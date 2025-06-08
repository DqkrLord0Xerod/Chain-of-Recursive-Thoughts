from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List

import tiktoken

from core.interfaces import LLMProvider, CacheProvider, QualityEvaluator
from core.context_manager import ContextManager
from core.resilience import RetryState, with_retry


def _cache_key(messages: List[Dict[str, str]]) -> str:
    raw = json.dumps(messages, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class ReasonerState:
    best_response: str = ""
    best_score: float = 0.0
    history: List[Dict[str, str]] = field(default_factory=list)


class ChatEngine:
    """Reasoning engine using dependency injection."""

    def __init__(
        self,
        llm: LLMProvider,
        cache: CacheProvider,
        evaluator: QualityEvaluator,
        *,
        max_tokens: int = 2000,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.evaluator = evaluator
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            from tiktoken import _educational

            self.tokenizer = _educational.train_simple_encoding()
        self.context = ContextManager(max_tokens, self.tokenizer)
        self.state = ReasonerState()

    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        key = _cache_key(messages)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        async def coro():
            return await self.llm.chat(messages, temperature=temperature)

        resp = await with_retry(coro, RetryState())
        self.cache.set(key, resp)
        return resp

    async def think(self, prompt: str, *, max_rounds: int = 3) -> str:
        messages = [{"role": "user", "content": prompt}]
        best = await self._call_llm(messages)
        self.state.best_response = best
        self.state.best_score = self.evaluator.score(best, prompt)
        self.state.history.append({"round": 0, "response": best})
        for round_no in range(1, max_rounds + 1):
            alternatives = 1 if self.state.best_score > 0.9 else 3
            tasks = []
            for i in range(alternatives):
                alt_prompt = f"Alternative {i+1}: {prompt}"
                msgs = [{"role": "user", "content": alt_prompt}]
                tasks.append(asyncio.create_task(self._call_llm(msgs)))
            done, _ = await asyncio.wait(tasks, timeout=10)
            if not done:
                break
            chosen = None
            for task in done:
                text = task.result()
                score = self.evaluator.score(text, prompt)
                self.state.history.append({"round": round_no, "response": text})
                if score > self.state.best_score:
                    self.state.best_score = score
                    self.state.best_response = text
                    chosen = text
            if chosen is None:
                break
        return self.state.best_response
