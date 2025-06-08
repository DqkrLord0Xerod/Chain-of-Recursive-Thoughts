import os
import json
import logging
import hashlib
import math
import re
from collections import OrderedDict
from typing import List, Dict, Callable, Tuple
import pickle
from datetime import datetime
import requests
import tiktoken
from tiktoken import _educational
import contextlib
import io
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvergenceTracker:
    """Track response quality to detect convergence or oscillation."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float],
        score_fn: Callable[[str, str], float],
        similarity_threshold: float = 0.95,
        quality_threshold: float = 0.01,
        oscillation_threshold: float = 0.95,
        history_size: int = 5,
    ) -> None:
        self.similarity_fn = similarity_fn
        self.score_fn = score_fn
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.oscillation_threshold = oscillation_threshold
        self.history_size = history_size
        self.history: List[Tuple[str, float]] = []

    def add(self, response: str, prompt: str) -> None:
        score = self.score_fn(response, prompt)
        self.history.append((response, score))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def should_continue(self, prompt: str) -> Tuple[bool, str]:
        if len(self.history) < 2:
            return True, "insufficient history"

        prev_resp, prev_score = self.history[-2]
        curr_resp, curr_score = self.history[-1]

        similarity = self.similarity_fn(prev_resp, curr_resp)
        if similarity >= self.similarity_threshold:
            return False, "converged"

        improvement = curr_score - prev_score
        if improvement < self.quality_threshold:
            return False, "quality plateau"

        for old_resp, _ in self.history[:-2]:
            if (
                self.similarity_fn(old_resp, curr_resp)
                >= self.oscillation_threshold
            ):
                return False, "oscillation"

        return True, "continue"


class QualityAssessor:
    """Compute simple quality metrics for responses."""

    def __init__(self, similarity_fn: Callable[[str, str], float]) -> None:
        self.similarity_fn = similarity_fn

    def relevance(self, prompt: str, response: str) -> float:
        return self.similarity_fn(prompt, response)

    def completeness(self, prompt: str, response: str) -> float:
        words_prompt = set(prompt.lower().split())
        words_resp = set(response.lower().split())
        if not words_prompt:
            return 0.0
        return len(words_prompt & words_resp) / len(words_prompt)

    def clarity(self, response: str) -> float:
        if not response:
            return 0.0
        sentences = re.split(r"[.!?]+", response)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_len = len(response.split()) / len(sentences)
        return max(0.0, 1.0 - (avg_len - 20) / 20)

    def accuracy(self, prompt: str, response: str) -> float:
        return self.similarity_fn(prompt, response)

    def comprehensive_score(self, response: str, prompt: str) -> Dict[str, float]:
        metrics = {
            "relevance": self.relevance(prompt, response),
            "completeness": self.completeness(prompt, response),
            "clarity": self.clarity(response),
            "accuracy": self.accuracy(prompt, response),
        }
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        return metrics


class ContextManager:
    """Manage pruning of conversation history to fit token limits."""

    def __init__(self, max_tokens: int, tokenizer) -> None:
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def optimize_context(self, messages: List[Dict]) -> List[Dict]:
        """Return a trimmed list of messages within the token budget."""
        if not messages:
            return []

        def msg_tokens(msg: Dict) -> int:
            return self._count(msg.get("content", ""))

        total = sum(msg_tokens(m) for m in messages)
        if total <= self.max_tokens:
            return list(messages)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        trimmed_sys: List[Dict] = []
        token_total = 0
        for msg in reversed(system_msgs):
            t = msg_tokens(msg)
            if token_total + t <= self.max_tokens:
                trimmed_sys.insert(0, msg)
                token_total += t

        non_system_trim: List[Dict] = []
        for msg in reversed(non_system):
            t = msg_tokens(msg)
            if token_total + t > self.max_tokens:
                break
            non_system_trim.insert(0, msg)
            token_total += t

        return trimmed_sys + non_system_trim


class EnhancedRecursiveThinkingChat:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
        max_context_tokens: int = 2000,
        caching_enabled: bool = True,
        cache_size: int = 128,
        max_retries: int = 3,
        disk_cache_path: str | None = None,
        disk_cache_size: int = 256,
    ) -> None:
        """Initialize with OpenRouter API.

        Args:
            api_key: The API key for OpenRouter.
            model: The model identifier.
            max_context_tokens: Maximum tokens to keep in history.
            caching_enabled: Enable the in-memory cache.
            cache_size: Maximum entries to store in the cache.
            max_retries: API retry attempts on failure.
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
        self.disk_cache_path = disk_cache_path
        self.disk_cache_size = disk_cache_size
        self.disk_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        if self.disk_cache_path:
            self._load_disk_cache()
        self.max_retries = max_retries
        self.quality_assessor = QualityAssessor(self._semantic_similarity)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.tokenizer = _educational.train_simple_encoding()
        self.context_manager = ContextManager(max_context_tokens, self.tokenizer)

    def _estimate_tokens(self, text: str) -> int:
        """Return the token count for the given text."""
        return len(self.tokenizer.encode(text))

    def _history_token_count(self) -> int:
        return sum(
            self._estimate_tokens(m.get("content", ""))
            for m in self.conversation_history
        )

    def _trim_conversation_history(self) -> None:
        """Trim conversation history using ``ContextManager``."""
        self.conversation_history = self.context_manager.optimize_context(
            self.conversation_history
        )

    def _load_disk_cache(self) -> None:
        """Load persistent cache entries from disk."""
        if not self.disk_cache_path or not os.path.exists(self.disk_cache_path):
            return
        try:
            with open(self.disk_cache_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.disk_cache = OrderedDict(data)
                while len(self.disk_cache) > self.disk_cache_size:
                    self.disk_cache.popitem(last=False)
                for k in list(self.disk_cache)[-self.cache_size:]:
                    self.cache[k] = self.disk_cache[k]
        except Exception as e:  # pragma: no cover - logging not tested
            logger.warning("Failed to load disk cache: %s", e)

    def _save_disk_cache(self) -> None:
        """Persist cache entries to disk."""
        if not self.disk_cache_path:
            return
        try:
            with open(self.disk_cache_path, "wb") as f:
                pickle.dump(self.disk_cache, f)
        except Exception as e:  # pragma: no cover - logging not tested
            logger.warning("Failed to save disk cache: %s", e)

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
        messages = self.context_manager.optimize_context(messages)
        cache_key = self._cache_key(messages)
        if self.caching_enabled:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if stream:
                    print(cached)
                self.cache.move_to_end(cache_key)
                return cached
            if self.disk_cache_path and cache_key in self.disk_cache:
                cached = self.disk_cache[cache_key]
                if stream:
                    print(cached)
                self.cache[cache_key] = cached
                self.cache.move_to_end(cache_key)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
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

        for attempt in range(1, self.max_retries + 1):
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
                            line = line.decode("utf-8")
                            if line.startswith("data: "):
                                line = line[6:]
                                if line.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(line)
                                    if (
                                        "choices" in chunk
                                        and len(chunk["choices"]) > 0
                                    ):
                                        delta = chunk["choices"][0].get(
                                            "delta", {}
                                        )
                                        content = delta.get("content", "")
                                        if content:
                                            full_response += content
                                            print(content, end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                    print()
                    result = full_response
                else:
                    result = (
                        response.json()["choices"][0]["message"]["content"]
                        .strip()
                    )
                if self.caching_enabled:
                    self.cache[cache_key] = result
                    self.cache.move_to_end(cache_key)
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                    if self.disk_cache_path:
                        self.disk_cache[cache_key] = result
                        self.disk_cache.move_to_end(cache_key)
                        while len(self.disk_cache) > self.disk_cache_size:
                            self.disk_cache.popitem(last=False)
                        self._save_disk_cache()
                return result
            except Exception as e:  # pragma: no cover - logging not tested
                logger.warning(
                    "API call failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt == self.max_retries:
                    return "Error: Could not get response from API"
                time.sleep(2 ** (attempt - 1))
    
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

    @staticmethod
    def _simple_overlap(text1: str, text2: str) -> float:

        """Return a simple lexical overlap score."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1:
            return 0.0
        return len(words1 & words2) / len(words1)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Return semantic similarity using embeddings with fallback."""
        if not text1 or not text2:
            return 0.0
        url = "https://openrouter.ai/api/v1/embeddings"
        payload = {
            "model": "openai/text-embedding-ada-002",
            "input": [text1, text2],
        }
        try:
            resp = requests.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            emb1 = data["data"][0]["embedding"]
            emb2 = data["data"][1]["embedding"]
            dot = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            if not norm1 or not norm2:
                return 0.0
            return dot / (norm1 * norm2)
        except Exception as e:
            logger.error("Embedding similarity failed: %s", e)
            return self._simple_overlap(text1, text2)

    def _score_response(self, response: str, prompt: str) -> float:
        """Return an overall quality score for ranking."""
        return self.quality_assessor.comprehensive_score(response, prompt)["overall"]

    def _batch_generate_and_evaluate(
        self,
        current_best: str,
        prompt: str,
        num_alternatives: int = 3,
    ) -> tuple[str, List[str], str]:
        """Generate alternatives and evaluate all options in one API call."""

        batch_prompt = (
            f"Original message: {prompt}\n\n"
            f"Current response: {current_best}\n\n"
            f"Generate {num_alternatives} alternative responses and then rate"
            " all options (including the current one) for accuracy, completeness,"
            " clarity, and relevance on a scale of 1-10. Respond in JSON like:"
            " {\"alternatives\": [\"alt1\", ...], \"current\": {\"accuracy\": 8,"
            " \"completeness\": 8, \"clarity\": 8, \"relevance\": 8}, \"1\": {..},"
            " \"choice\": \"1\", \"reason\": \"text\"}"
        )

        messages = self.conversation_history + [
            {"role": "user", "content": batch_prompt}
        ]

        raw = self._call_api(messages, temperature=0.7, stream=True)

        data: Dict | None = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None

        alternatives: List[str] = []
        choice = "current"
        explanation = "No explanation provided"

        if isinstance(data, dict):
            alts = data.get("alternatives", [])
            if isinstance(alts, list):
                alternatives = [str(a).strip() for a in alts][:num_alternatives]
            choice = str(data.get("choice", choice)).lower()
            explanation = data.get("reason", explanation)
            score_map = {k: v for k, v in data.items() if isinstance(v, dict)}
        else:
            lines = [line.strip() for line in raw.split("\n") if line.strip()]
            alternatives = lines[:num_alternatives]
            score_map = {}

        responses = [current_best] + alternatives
        labels = ["current"] + [str(i + 1) for i in range(len(alternatives))]

        scores = []
        for label, resp in zip(labels, responses):
            base = self.quality_assessor.comprehensive_score(resp, prompt)["overall"]
            metrics = score_map.get(label, {})
            model_score = sum(
                float(metrics.get(m, 0))
                for m in ("accuracy", "completeness", "clarity", "relevance")
            )
            model_score /= 40.0
            scores.append(base + model_score)

        try:
            model_index = 0 if choice == "current" else int(choice)
        except Exception:
            model_index = None

        if model_index is not None and 0 <= model_index < len(scores):
            scores[model_index] += 1.0

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_response = responses[best_idx]

        return best_response, alternatives, explanation

    def _should_continue_thinking(
        self,
        previous_response: str,
        new_response: str,
        prompt: str,
        similarity_threshold: float = 0.95,
        quality_threshold: float = 0.01,
    ) -> bool:
        """Return ``True`` if another round is likely beneficial."""
        if not previous_response or not new_response:
            return False

        similarity = self._semantic_similarity(previous_response, new_response)
        if similarity >= similarity_threshold:
            return False

        old_score = self.quality_assessor.comprehensive_score(
            previous_response, prompt
        )["overall"]
        new_score = self.quality_assessor.comprehensive_score(
            new_response, prompt
        )["overall"]
        return (new_score - old_score) >= quality_threshold

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
        messages = self.conversation_history + [
            {"role": "user", "content": user_input}
        ]
        current_best = self._call_api(messages, stream=True)
        print("=" * 50)

        thinking_history = [
            {"round": 0, "response": current_best, "selected": True}
        ]

        tracker = ConvergenceTracker(
            self._semantic_similarity,
            lambda resp, prompt: self.quality_assessor.comprehensive_score(
                resp, prompt
            )["overall"],
        )
        tracker.add(current_best, user_input)
        
        # Iterative improvement
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                print(f"\n=== ROUND {round_num}/{thinking_rounds} ===")
            
            # Generate alternatives and evaluate in one call
            new_best, alternatives, explanation = self._batch_generate_and_evaluate(
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
                    "alternative_number": i + 1,
                })

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

            tracker.add(current_best, user_input)
            cont, _ = tracker.should_continue(user_input)
            if not cont:
                break
        
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
