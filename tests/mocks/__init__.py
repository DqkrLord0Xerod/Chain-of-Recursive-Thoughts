from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator

from core.interfaces import LLMProvider, CacheProvider, QualityEvaluator


@dataclass
class MockLLMResponse:
    content: str
    usage: Dict[str, int]
    model: str
    cached: bool = False


class MockLLMProvider(LLMProvider):
    """Simple LLM provider returning predefined responses."""

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = responses or ["ok"]
        self.call_count = 0
        self.model = "mock"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> MockLLMResponse:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        tokens = len(response.split())
        return MockLLMResponse(
            content=response,
            usage={"total_tokens": tokens},
            model=self.model,
            cached=False,
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        for chunk in response.split():
            yield chunk


class MockCacheProvider(CacheProvider):
    """In-memory cache provider for testing."""

    def __init__(self) -> None:
        self.store: Dict[str, Any] = {}

    async def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def clear(self, *, tag: Optional[str] = None) -> int:
        count = len(self.store)
        self.store.clear()
        return count

    async def stats(self) -> Dict[str, Any]:
        return {"size": len(self.store)}


class MockQualityEvaluator(QualityEvaluator):
    """Quality evaluator returning preset scores."""

    def __init__(self, scores: Optional[Dict[str, float]] = None, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.scores = scores or {}
        self.thresholds = thresholds or {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return self.scores.get(response, 0.5)

    def detailed_score(self, response: str, prompt: str) -> Dict[str, float]:
        return {"overall": self.score(response, prompt)}
