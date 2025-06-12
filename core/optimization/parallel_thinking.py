"""Stub implementations for optimization classes used in tests."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple


class ThinkingCandidate:
    def __init__(self, response: str, quality_score: float = 0.0) -> None:
        self.response = response
        self.quality_score = quality_score
        self.generation_time = 0.0
        self.tokens_used = 0
        self.source = "stub"
        self.metadata: Dict[str, Any] = {}


class ParallelThinkingOptimizer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def think_parallel(
        self,
        prompt: str,
        initial_response: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, List[ThinkingCandidate], Dict[str, Any]]:
        return initial_response, [], {"rounds": 0}


class AdaptiveThinkingOptimizer:
    def __init__(self, parallel_optimizer: ParallelThinkingOptimizer, *args: Any, **kwargs: Any) -> None:
        self.parallel_optimizer = parallel_optimizer

    async def think_adaptive(
        self,
        prompt: str,
        initial_response: str,
        category: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, List[ThinkingCandidate], Dict[str, Any]]:
        return await self.parallel_optimizer.think_parallel(prompt, initial_response)
