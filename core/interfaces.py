from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Dict


@dataclass
class Message:
    role: str
    content: str


class LLMProvider(Protocol):
    """Interface for large language model providers."""

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        """Send chat completion request and return response text."""


class CacheProvider(Protocol):
    """Interface for caching backend."""

    def get(self, key: str) -> str | None:
        """Retrieve value from cache."""

    def set(self, key: str, value: str) -> None:
        """Store value in cache."""


class QualityEvaluator(Protocol):
    """Interface for scoring model responses."""

    def score(self, response: str, prompt: str) -> float:
        """Return quality score between 0 and 1."""
