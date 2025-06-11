from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import aiohttp
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Dict, List, Tuple


import structlog
import tiktoken
from tiktoken import _educational

from config import settings
from core.context import ContextManager
from core.recursion import ConvergenceTracker, QualityAssessor
from monitoring import MetricsRecorder
from core.llm_client import LLMClient
from core.cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())


@dataclass
class ThinkingResult:
    response: str
    thinking_rounds: int
    thinking_history: List[Dict]
    api_calls: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class CoRTConfig:
    """Configuration for :class:`EnhancedRecursiveThinkingChat`."""

    api_key: str | None = field(default_factory=lambda: settings.openrouter_api_key)
    model: str = field(default_factory=lambda: settings.model)
    max_context_tokens: int = 2000
    caching_enabled: bool = True
    cache_size: int = 128
    max_retries: int = 3
    disk_cache_path: str | None = None
    disk_cache_size: int = 256

    @classmethod
    def from_file(cls, path: str) -> "CoRTConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class EnhancedRecursiveThinkingChat:
    def __init__(self, config: CoRTConfig) -> None:
        self.llm_client = LLMClient(
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
            model=config.model,
            max_retries=config.max_retries,
        )
        self.cache_manager = CacheManager(
            enabled=config.caching_enabled,
            memory_size=config.cache_size,
            disk_path=config.disk_cache_path,
            disk_size=config.disk_cache_size,
        )
        self.max_context_tokens = config.max_context_tokens
        self.conversation_history: List[Dict] = []
        self.full_thinking_log: List[Dict] = []
        self.quality_assessor = QualityAssessor(self._semantic_similarity)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.tokenizer = _educational.train_simple_encoding()
        self.context_manager = ContextManager(
            config.max_context_tokens, self.tokenizer, self._summarize_messages
        )
        self.logger = structlog.get_logger(__name__)

    # ------------------------------------
    # Internal helpers
    # ------------------------------------
    def _cache_key(self, messages: List[Dict]) -> Tuple[str, str]:
        if not messages:
            return "", ""
        prompt = messages[-1].get("content", "")
        context = json.dumps(messages[:-1], sort_keys=True)
        context_hash = hashlib.md5(context.encode("utf-8")).hexdigest()
        return prompt, context_hash

    def _trim_conversation_history(self) -> None:
        self.conversation_history = self.context_manager.optimize_context(self.conversation_history)

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """Summarize a list of messages preserving decisions and intent."""
        content = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        prompt = (
            "Summarize the following conversation in a concise form while "
            "preserving intent and prior decisions:\n" + content
        )
        return self.llm_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False,
        )

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _calculate_token_usage(self, api_calls: List[Dict]) -> int:
        total = 0
        for call in api_calls:
            for msg in call.get("messages", []):
                total += self._token_count(msg.get("content", ""))
            total += self._token_count(call.get("response", ""))
        return total

    def _call_api(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        messages = self.context_manager.optimize_context(messages)
        cache_key = self._cache_key(messages)
        api_entry = {"messages": messages}
        cached = self.cache_manager.get(cache_key)
        if cached is not None:
            if stream:
                self.logger.info(cached)
            api_entry["response"] = cached
            self.full_thinking_log.append(api_entry)
            return cached

        result = self.llm_client.chat(
            messages,
            temperature=temperature,
            stream=stream,
        )
        self.cache_manager.set(cache_key, result)
        api_entry["response"] = result
        self.full_thinking_log.append(api_entry)
        return result

    async def _async_call_api(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
    ) -> str:
        messages = self.context_manager.optimize_context(messages)
        cache_key = self._cache_key(messages)
        api_entry = {"messages": messages}
        cached = self.cache_manager.get(cache_key)
        if cached is not None:
            api_entry["response"] = cached
            self.full_thinking_log.append(api_entry)
            return cached

        async with self.semaphore:
            result = await self.llm_client.async_chat(
                messages,
                temperature=temperature,
            )
        self.cache_manager.set(cache_key, result)
        api_entry["response"] = result
        self.full_thinking_log.append(api_entry)
        return result

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        try:
            emb1, emb2 = self.llm_client.embeddings([text1, text2])
            dot = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            if not norm1 or not norm2:
                return 0.0
            return dot / (norm1 * norm2)
        except Exception as e:
            self.logger.error("Embedding similarity failed: %s", e)
            return self._simple_overlap(text1, text2)

    @staticmethod
    def _simple_overlap(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1:
            return 0.0
        return len(words1 & words2) / len(words1)

    # ------------------------------------
    # Core algorithm
    # ------------------------------------
    def _determine_thinking_rounds(self, prompt: str) -> int:
        meta_prompt = f"""Given this message: "{prompt}"

How many rounds of iterative thinking (1-5) would be optimal to generate the best response?
Consider the complexity and nuance required.
Respond with just a number between 1 and 5."""
        messages = [{"role": "user", "content": meta_prompt}]
        self.logger.info("=== DETERMINING THINKING ROUNDS ===")
        response = self._call_api(messages, temperature=0.3, stream=True)
        self.logger.info("=" * 50)
        try:
            rounds = int("".join(filter(str.isdigit, response)))
            return min(max(rounds, 1), 5)
        except (ValueError, TypeError) as e:
            self.logger.error(
                "Failed to parse thinking rounds from API response: %s", e
            )
            return 3

    def _batch_generate_and_evaluate(
        self,
        current_best: str,
        prompt: str,
        num_alternatives: int = 3,
    ) -> Tuple[str, List[str], str]:
        batch_prompt = (
            f"Original message: {prompt}\n\n"
            f"Current response: {current_best}\n\n"
            f"Generate {num_alternatives} alternative responses and then rate"
            " all options (including the current one) for accuracy, completeness,"
            " clarity, and relevance on a scale of 1-10. Respond in JSON like:"
            " {\"alternatives\": [\"alt1\", ...], \"current\": {\"accuracy\": 8,"
            " \"completeness\": 8, \"clarity\": 8, \"relevance\": 8}, \"1\": {...},"
            " \"choice\": \"1\", \"reason\": \"text\"}"
        )
        messages = self.conversation_history + [{"role": "user", "content": batch_prompt}]
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

    def think_and_respond(
        self,
        user_input: str,
        verbose: bool = True,
        thinking_rounds: int | None = None,
        alternatives_per_round: int = 3,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> ThinkingResult:
        self.logger.info("=" * 50)
        self.logger.info("🤔 RECURSIVE THINKING PROCESS STARTING")
        self.logger.info("=" * 50)
        start_index = len(self.full_thinking_log)
        start_time = time.time()
        if thinking_rounds is None:
            thinking_rounds = self._determine_thinking_rounds(user_input)
        if verbose:
            self.logger.info("🤔 Thinking... (%s rounds needed)", thinking_rounds)
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        current_best = self._call_api(messages, stream=True)
        self.logger.info("=" * 50)
        thinking_history = [{"round": 0, "response": current_best, "selected": True}]
        tracker = ConvergenceTracker(
            self._semantic_similarity,
            lambda resp, p: self.quality_assessor.comprehensive_score(resp, p)["overall"],
        )
        tracker.add(current_best, user_input)
        convergence_reason = "max_rounds"
        rounds_completed = 0
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                self.logger.info("=== ROUND %s/%s ===", round_num, thinking_rounds)
            new_best, alternatives, explanation = self._batch_generate_and_evaluate(
                current_best,
                user_input,
                alternatives_per_round,
            )
            for i, alt in enumerate(alternatives):
                thinking_history.append(
                    {
                        "round": round_num,
                        "response": alt,
                        "selected": False,
                        "alternative_number": i + 1,
                    }
                )
            if new_best != current_best:
                for item in thinking_history:
                    if item["round"] == round_num and item["response"] == new_best:
                        item["selected"] = True
                        item["explanation"] = explanation
                current_best = new_best
                if verbose:
                    self.logger.info("    \u2713 Selected alternative: %s", explanation)
            else:
                for item in thinking_history:
                    if item["selected"] and item["response"] == current_best:
                        item["explanation"] = explanation
                        break
                if verbose:
                    self.logger.info("    \u2713 Kept current response: %s", explanation)
            tracker.add(current_best, user_input)
            cont, reason = tracker.should_continue(user_input)
            rounds_completed += 1
            if not cont:
                convergence_reason = reason
                break
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": current_best})
        self.conversation_history = self.context_manager.optimize_context(self.conversation_history)
        self.logger.info("=" * 50)
        self.logger.info("🎯 FINAL RESPONSE SELECTED")
        self.logger.info("=" * 50)
        api_calls = self.full_thinking_log[start_index:]
        processing_time = time.time() - start_time
        token_usage = self._calculate_token_usage(api_calls)
        if metrics_recorder is not None:
            metrics_recorder.record_run(
                processing_time=processing_time,
                token_usage=token_usage,
                num_rounds=rounds_completed,
                convergence_reason=convergence_reason,
            )
        return ThinkingResult(
            response=current_best,
            thinking_rounds=thinking_rounds,
            thinking_history=thinking_history,
            api_calls=api_calls,
            processing_time=processing_time,
        )

    def save_full_log(self, filename: str | None = None) -> None:
        if filename is None:
            filename = f"full_thinking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conversation": self.conversation_history,
                    "full_thinking_log": self.full_thinking_log,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info("Full thinking log saved to %s", filename)

    def save_conversation(self, filename: str | None = None) -> None:
        if filename is None:
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conversation": self.conversation_history,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info("Conversation saved to %s", filename)


class AsyncEnhancedRecursiveThinkingChat(EnhancedRecursiveThinkingChat):
    def __init__(self, config: CoRTConfig, max_connections: int = 5) -> None:
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(max_connections)

    async def __aenter__(self) -> "AsyncEnhancedRecursiveThinkingChat":
        if self.llm_client.session is None:
            self.llm_client.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.llm_client.close()

    async def _parallel_alternative_generation(
        self,
        current_best: str,
        prompt: str,
        num_alternatives: int = 3,
        tracker: ConvergenceTracker | None = None,
    ) -> AsyncIterator[str]:
        if tracker is None:
            tracker = ConvergenceTracker(
                self._semantic_similarity,
                lambda r, p: self.quality_assessor.comprehensive_score(r, p)["overall"],
            )
        tracker.add(current_best, prompt)
        for idx in range(num_alternatives):
            alt_prompt = f"Alternative #{idx + 1} for '{prompt}' based on '{current_best}'."
            messages = self.conversation_history + [{"role": "user", "content": alt_prompt}]
            alt = await self._async_call_api(messages, temperature=0.7)
            cont, _ = tracker.update(alt, prompt)
            yield alt
            if not cont:
                break

    async def _async_determine_thinking_rounds(self, prompt: str) -> int:
        meta_prompt = f"""Given this message: "{prompt}"

How many rounds of iterative thinking (1-5) would be optimal to generate the best response?
Consider the complexity and nuance required.
Respond with just a number between 1 and 5."""
        messages = [{"role": "user", "content": meta_prompt}]
        self.logger.info("=== DETERMINING THINKING ROUNDS ===")
        response = await self._async_call_api(messages, temperature=0.3)
        self.logger.info("=" * 50)
        try:
            rounds = int("".join(filter(str.isdigit, response)))
            return min(max(rounds, 1), 5)
        except (ValueError, TypeError) as e:
            self.logger.error("Failed to parse thinking rounds from API response: %s", e)
            return 3

    async def _async_batch_generate_and_evaluate(
        self,
        current_best: str,
        prompt: str,
        num_alternatives: int = 3,
    ) -> Tuple[str, List[str], str]:
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
        messages = self.conversation_history + [{"role": "user", "content": batch_prompt}]
        raw = await self._async_call_api(messages, temperature=0.7)
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
                float(metrics.get(m, 0)) for m in ("accuracy", "completeness", "clarity", "relevance")
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

    async def think_and_respond(
        self,
        user_input: str,
        verbose: bool = True,
        thinking_rounds: int | None = None,
        alternatives_per_round: int = 3,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> ThinkingResult:
        self.logger.info("=" * 50)
        self.logger.info("🤔 RECURSIVE THINKING PROCESS STARTING")
        self.logger.info("=" * 50)
        start_index = len(self.full_thinking_log)
        start_time = time.time()
        if thinking_rounds is None:
            thinking_rounds = await self._async_determine_thinking_rounds(user_input)
        if verbose:
            self.logger.info("🤔 Thinking... (%s rounds needed)", thinking_rounds)
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        current_best = await self._async_call_api(messages)
        self.logger.info("=" * 50)
        thinking_history = [{"round": 0, "response": current_best, "selected": True}]
        tracker = ConvergenceTracker(
            self._semantic_similarity,
            lambda resp, p: self.quality_assessor.comprehensive_score(resp, p)["overall"],
        )
        tracker.add(current_best, user_input)
        convergence_reason = "max_rounds"
        rounds_completed = 0
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                self.logger.info("=== ROUND %s/%s ===", round_num, thinking_rounds)
            new_best, alternatives, explanation = await self._async_batch_generate_and_evaluate(
                current_best,
                user_input,
                alternatives_per_round,
            )
            for i, alt in enumerate(alternatives):
                thinking_history.append(
                    {
                        "round": round_num,
                        "response": alt,
                        "selected": False,
                        "alternative_number": i + 1,
                    }
                )
            if new_best != current_best:
                for item in thinking_history:
                    if item["round"] == round_num and item["response"] == new_best:
                        item["selected"] = True
                        item["explanation"] = explanation
                current_best = new_best
                if verbose:
                    self.logger.info("    \u2713 Selected alternative: %s", explanation)
            else:
                for item in thinking_history:
                    if item["selected"] and item["response"] == current_best:
                        item["explanation"] = explanation
                        break
                if verbose:
                    self.logger.info("    \u2713 Kept current response: %s", explanation)
            tracker.add(current_best, user_input)
            cont, reason = tracker.should_continue(user_input)
            rounds_completed += 1
            if not cont:
                convergence_reason = reason
                break
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": current_best})
        self.conversation_history = self.context_manager.optimize_context(self.conversation_history)
        self.logger.info("=" * 50)
        self.logger.info("🎯 FINAL RESPONSE SELECTED")
        self.logger.info("=" * 50)
        api_calls = self.full_thinking_log[start_index:]
        processing_time = time.time() - start_time
        token_usage = self._calculate_token_usage(api_calls)
        if metrics_recorder is not None:
            metrics_recorder.record_run(
                processing_time=processing_time,
                token_usage=token_usage,
                num_rounds=rounds_completed,
                convergence_reason=convergence_reason,
            )
        return ThinkingResult(
            response=current_best,
            thinking_rounds=thinking_rounds,
            thinking_history=thinking_history,
            api_calls=api_calls,
            processing_time=processing_time,
        )

    async def stream_think_and_respond(
        self,
        user_input: str,
        verbose: bool = True,
        thinking_rounds: int | None = None,
        alternatives_per_round: int = 3,
        metrics_recorder: MetricsRecorder | None = None,
    ) -> AsyncIterator[Dict]:
        """Yield progressive updates while thinking."""

        start_index = len(self.full_thinking_log)
        start_time = time.time()

        if thinking_rounds is None:
            thinking_rounds = await self._async_determine_thinking_rounds(user_input)

        if verbose:
            self.logger.info("🤔 Thinking... (%s rounds needed)", thinking_rounds)

        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        current_best = await self._async_call_api(messages)
        yield {"round": 0, "response": current_best, "selected": True}

        thinking_history = [{"round": 0, "response": current_best, "selected": True}]

        tracker = ConvergenceTracker(
            self._semantic_similarity,
            lambda resp, p: self.quality_assessor.comprehensive_score(resp, p)["overall"],
        )
        tracker.add(current_best, user_input)

        convergence_reason = "max_rounds"
        rounds_completed = 0

        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                self.logger.info("=== ROUND %s/%s ===", round_num, thinking_rounds)

            new_best, alternatives, explanation = await self._async_batch_generate_and_evaluate(
                current_best,
                user_input,
                alternatives_per_round,
            )

            for i, alt in enumerate(alternatives):
                entry = {
                    "round": round_num,
                    "response": alt,
                    "selected": False,
                    "alternative_number": i + 1,
                }
                thinking_history.append(entry)
                yield entry

            if new_best != current_best:
                for item in thinking_history:
                    if item["round"] == round_num and item["response"] == new_best:
                        item["selected"] = True
                        item["explanation"] = explanation
                current_best = new_best
            else:
                for item in thinking_history:
                    if item["selected"] and item["response"] == current_best:
                        item["explanation"] = explanation
                        break

            tracker.add(current_best, user_input)
            cont, reason = tracker.should_continue(user_input)
            rounds_completed += 1
            if not cont:
                convergence_reason = reason
                break

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": current_best})
        self.conversation_history = self.context_manager.optimize_context(self.conversation_history)

        api_calls = self.full_thinking_log[start_index:]
        processing_time = time.time() - start_time
        token_usage = self._calculate_token_usage(api_calls)

        if metrics_recorder is not None:
            metrics_recorder.record_run(
                processing_time=processing_time,
                token_usage=token_usage,
                num_rounds=rounds_completed,
                convergence_reason=convergence_reason,
            )

        result = ThinkingResult(
            response=current_best,
            thinking_rounds=thinking_rounds,
            thinking_history=thinking_history,
            api_calls=api_calls,
            processing_time=processing_time,
        )

        yield {
            "final": True,
            "response": result.response,
            "thinking_rounds": result.thinking_rounds,
            "thinking_history": result.thinking_history,
        }
