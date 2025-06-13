from __future__ import annotations

from core.interfaces import LLMProvider
from .base import ThinkingStrategy, QualityEvaluator

from .adaptive import AdaptiveThinkingStrategy
from .fixed import FixedThinkingStrategy
from .hybrid import HybridToolStrategy


_STRATEGY_MAP = {
    "adaptive": AdaptiveThinkingStrategy,
    "fixed": FixedThinkingStrategy,
    "hybrid": HybridToolStrategy,
}


class StrategyFactory:
    """Factory for creating thinking strategies."""

    def __init__(self, llm: LLMProvider, evaluator: QualityEvaluator) -> None:
        self.llm = llm
        self.evaluator = evaluator
        self._registry = dict(_STRATEGY_MAP)

    def register(self, name: str, cls: type[ThinkingStrategy]) -> None:
        """Register a new strategy class."""
        self._registry[name.lower()] = cls

    def create(self, name: str, **kwargs) -> ThinkingStrategy:
        """Instantiate a strategy by name."""
        cls = self._registry.get(name.lower(), AdaptiveThinkingStrategy)
        if cls is FixedThinkingStrategy:
            return cls(**kwargs)
        return cls(self.llm, self.evaluator, **kwargs)


def load_strategy(
    name: str,
    llm: LLMProvider,
    evaluator: QualityEvaluator,
    **kwargs,
) -> ThinkingStrategy:
    """Compatibility wrapper around :class:`StrategyFactory`."""
    return StrategyFactory(llm, evaluator).create(name, **kwargs)
