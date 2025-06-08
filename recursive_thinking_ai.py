import os
import json
import logging
import hashlib
from collections import OrderedDict
from typing import List, Dict
from datetime import datetime
import requests
import tiktoken
from tiktoken import _educational
import contextlib
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedRecursiveThinkingChat:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
        max_context_tokens: int = 2000,
        caching_enabled: bool = True,
        cache_size: int = 128,
    ) -> None:
        """Initialize with OpenRouter API.

        Args:
            api_key: The API key for OpenRouter.
            model: The model identifier.
            max_context_tokens: Maximum tokens to keep in history.
            caching_enabled: Enable the in-memory cache.
            cache_size: Maximum entries to store in the cache.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Recursive Thinking Chat",
            "Content-Type": "application/json",
        }
        self.conversation_history: List[Dict] = []
        self.full_thinking_log: List[Dict] = []
        self.caching_enabled = caching_enabled
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.tokenizer = _educational.train_simple_encoding()

    def _estimate_tokens(self, text: str) -> int:
        """Return the token count for the given text."""
        return len(self.tokenizer.encode(text))

    def _history_token_count(self) -> int:
        return sum(
            self._estimate_tokens(m.get("content", ""))
            for m in self.conversation_history
        )

    def _trim_conversation_history(self) -> None:
        """Trim conversation history to fit within ``max_context_tokens``."""
        while self._history_token_count() > self.max_context_tokens:
            if not self.conversation_history:
                break
            self.conversation_history.pop(0)

    def _cache_key(self, messages: List[Dict]) -> tuple[str, str]:
        """Return a cache key based on prompt and context hash."""
        if not messages:
            return "", ""
        prompt = messages[-1].get("content", "")
        context = json.dumps(messages[:-1], sort_keys=True)
        context_hash = hashlib.md5(context.encode("utf-8")).hexdigest()
        return prompt, context_hash

    def _call_api(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        """Make an API call to OpenRouter with optional caching."""
        cache_key = self._cache_key(messages)
        if self.caching_enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            if stream:
                print(cached)
            # Move key to end to mark as recently used
            self.cache.move_to_end(cache_key)
            return cached
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "reasoning": {
                "max_tokens": 10386,
            }
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                stream=stream,
            )
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            line = line[6:]
                            if line.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_response += content
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                continue
                print()  # New line after streaming
                result = full_response
            else:
                result = response.json()['choices'][0]['message']['content'].strip()
            if self.caching_enabled:
                self.cache[cache_key] = result
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            return result
        except Exception as e:
            print(f"API Error: {e}")
            return "Error: Could not get response from API"
    
    def _determine_thinking_rounds(self, prompt: str) -> int:
        """Let the model decide how many rounds of thinking are needed."""
        meta_prompt = f"""Given this message: "{prompt}"
        
How many rounds of iterative thinking (1-5) would be optimal to generate the best response?
Consider the complexity and nuance required.
Respond with just a number between 1 and 5."""
        
        messages = [{"role": "user", "content": meta_prompt}]
        
        print("\n=== DETERMINING THINKING ROUNDS ===")
        response = self._call_api(messages, temperature=0.3, stream=True)
        print("=" * 50 + "\n")
        
        try:
            rounds = int(''.join(filter(str.isdigit, response)))
            return min(max(rounds, 1), 5)
        except (ValueError, TypeError) as e:
            logger.error(
                "Failed to parse thinking rounds from API response: %s", e
            )
            return 3
    
    def _generate_alternatives(
        self, base_response: str, prompt: str, num_alternatives: int = 3
    ) -> List[str]:
        """Generate alternative responses with a single API request."""

        alt_prompt = (
            f"Original message: {prompt}\n\n"
            f"Current response: {base_response}\n\n"
            f"Generate {num_alternatives} alternative responses that might be"
            " better. Respond in JSON as {\"alternatives\": [\"alt1\","
            " \"alt2\", ...]}"
        )

        messages = self.conversation_history + [
            {"role": "user", "content": alt_prompt}
        ]
        raw_result = self._call_api(messages, temperature=0.7, stream=True)

        try:
            data = json.loads(raw_result)
            alternatives = data.get("alternatives", [])
            if isinstance(alternatives, list):
                return [
                    str(a).strip() for a in alternatives
                ][:num_alternatives]
        except json.JSONDecodeError:
            pass

        return [
            line.strip() for line in raw_result.split("\n") if line.strip()
        ][:num_alternatives]

    def _score_response(self, response: str, prompt: str) -> float:
        """Return a simple overlap score between the prompt and the response.

        The score is the fraction of prompt words that also appear in the
        candidate response. It ranges from 0.0 to 1.0.
        """
        prompt_words = set(prompt.lower().split())
        resp_words = set(response.lower().split())
        if not prompt_words:
            return 0.0
        return len(prompt_words & resp_words) / len(prompt_words)
    
    def _evaluate_responses(self, prompt: str, current_best: str, alternatives: List[str]) -> tuple[str, str]:
        """Evaluate responses and select the best one."""
        print("\n=== EVALUATING RESPONSES ===")
        eval_prompt = f"""Original message: {prompt}

Evaluate these responses and choose the best one:

Current best: {current_best}

Alternatives:
{chr(10).join([f"{i + 1}. {alt}" for i, alt in enumerate(alternatives)])}

Which response best addresses the original message? Consider accuracy, clarity, and completeness.
First, respond with ONLY 'current' or a number (1-{len(alternatives)}).
Then on a new line, explain your choice in one sentence."""
        
        messages = [{"role": "user", "content": eval_prompt}]
        evaluation = self._call_api(messages, temperature=0.2, stream=True)
        print("=" * 50)
        
        # Better parsing
        lines = [line.strip() for line in evaluation.split('\n') if line.strip()]
        
        choice = 'current'
        explanation = "No explanation provided"
        
        if lines:
            first_line = lines[0].lower()
            if 'current' in first_line:
                choice = 'current'
            else:
                for char in first_line:
                    if char.isdigit():
                        choice = char
                        break
            
            if len(lines) > 1:
                explanation = ' '.join(lines[1:])
        
        # Compute a heuristic score for the current response and all
        # alternatives based on word overlap with the prompt.
        scores = [self._score_response(current_best, prompt)] + [
            self._score_response(a, prompt) for a in alternatives
        ]

        try:
            model_index = 0 if choice == "current" else int(choice)
        except Exception:
            model_index = 0

        # Boost the score of the response chosen by the model.
        if 0 <= model_index < len(scores):
            scores[model_index] += 1.0

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        if best_idx == 0:
            return current_best, explanation
        return alternatives[best_idx - 1], explanation
    
    def think_and_respond(
        self,
        user_input: str,
        verbose: bool = True,
        thinking_rounds: int | None = None,
        alternatives_per_round: int = 3,
    ) -> Dict:
        """Process user input with recursive thinking.

        Args:
            user_input: The message from the user.
            verbose: Whether to print progress information.
            thinking_rounds: Number of thinking rounds to run. If ``None``,
                the model will determine the count automatically.
            alternatives_per_round: Number of alternative responses generated
                in each round.
        """
        print("\n" + "=" * 50)
        print("🤔 RECURSIVE THINKING PROCESS STARTING")
        print("=" * 50)
        
        if thinking_rounds is None:
            thinking_rounds = self._determine_thinking_rounds(user_input)
        
        if verbose:
            print(f"\n🤔 Thinking... ({thinking_rounds} rounds needed)")
        
        # Initial response
        print("\n=== GENERATING INITIAL RESPONSE ===")
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        current_best = self._call_api(messages, stream=True)
        print("=" * 50)
        
        thinking_history = [{"round": 0, "response": current_best, "selected": True}]
        
        # Iterative improvement
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                print(f"\n=== ROUND {round_num}/{thinking_rounds} ===")
            
            # Generate alternatives
            alternatives = self._generate_alternatives(
                current_best,
                user_input,
                alternatives_per_round,
            )
            
            # Store alternatives in history
            for i, alt in enumerate(alternatives):
                thinking_history.append({
                    "round": round_num,
                    "response": alt,
                    "selected": False,
                    "alternative_number": i + 1
                })
            
            # Evaluate and select best
            new_best, explanation = self._evaluate_responses(user_input, current_best, alternatives)
            
            # Update selection in history
            if new_best != current_best:
                for item in thinking_history:
                    if item["round"] == round_num and item["response"] == new_best:
                        item["selected"] = True
                        item["explanation"] = explanation
                current_best = new_best
                
                if verbose:
                    print(f"\n    ✓ Selected alternative: {explanation}")
            else:
                for item in thinking_history:
                    if item["selected"] and item["response"] == current_best:
                        item["explanation"] = explanation
                        break
                
                if verbose:
                    print(f"\n    ✓ Kept current response: {explanation}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": current_best})
        
        # Keep conversation history within the configured token limit
        self._trim_conversation_history()
        
        print("\n" + "=" * 50)
        print("🎯 FINAL RESPONSE SELECTED")
        print("=" * 50)
        
        return {
            "response": current_best,
            "thinking_rounds": thinking_rounds,
            "thinking_history": thinking_history
        }
    
    def save_full_log(self, filename: str = None):
        """Save the full thinking process log."""
        if filename is None:
            filename = f"full_thinking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation": self.conversation_history,
                "full_thinking_log": self.full_thinking_log,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Full thinking log saved to {filename}")
    
    def save_conversation(self, filename: str = None):
        """Save the conversation and thinking history."""
        if filename is None:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation": self.conversation_history,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Conversation saved to {filename}")

def main():
    print("🤖 Enhanced Recursive Thinking Chat")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your OpenRouter API key (or press Enter to use env variable): ").strip()
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: No API key provided and OPENROUTER_API_KEY not found in environment")
            return
    
    # Initialize chat
    chat = EnhancedRecursiveThinkingChat(api_key=api_key)
    
    print("\nChat initialized! Type 'exit' to quit, 'save' to save conversation.")
    print("The AI will think recursively before each response.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'save':
            chat.save_conversation()
            continue
        elif user_input.lower() == 'save full':
            chat.save_full_log()
            continue
        elif not user_input:
            continue
        
        # Get response with thinking process
        result = chat.think_and_respond(user_input)
        
        print(f"\n🤖 AI FINAL RESPONSE: {result['response']}\n")
        
        # Always show complete thinking process
        print("\n--- COMPLETE THINKING PROCESS ---")
        for item in result['thinking_history']:
            print(f"\nRound {item['round']} {'[SELECTED]' if item['selected'] else '[ALTERNATIVE]'}:")
            print(f"  Response: {item['response']}")
            if 'explanation' in item and item['selected']:
                print(f"  Reason for selection: {item['explanation']}")
            print("-" * 50)
        print("--------------------------------\n")
    
    # Save on exit
    save_on_exit = input("Save conversation before exiting? (y/n): ").strip().lower()
    if save_on_exit == 'y':
        chat.save_conversation()
        save_full = input("Save full thinking log? (y/n): ").strip().lower()
        if save_full == 'y':
            chat.save_full_log()
    
    print("Goodbye! 👋")

if __name__ == "__main__":
    main()
