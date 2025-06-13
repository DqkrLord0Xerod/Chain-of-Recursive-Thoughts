from __future__ import annotations

from typing import TYPE_CHECKING

from core.interfaces import LLMProvider
from .base import ThinkingStrategy, QualityEvaluator

from .adaptive import AdaptiveThinkingStrategy
from .fixed import FixedThinkingStrategy
from . import get_strategy, register_strategy

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from core.chat_v2 import CoRTConfig


class StrategyFactory:
    """Factory for creating thinking strategies."""

    def __init__(self, llm: LLMProvider, evaluator: QualityEvaluator) -> None:
        self.llm = llm
        self.evaluator = evaluator

    def register(self, name: str, cls: type[ThinkingStrategy]) -> None:
        """Register a new strategy class globally."""
        register_strategy(name, cls)

    def create(self, name: str, **kwargs) -> ThinkingStrategy:
        """Instantiate a strategy by name."""
        cls = get_strategy(name) or AdaptiveThinkingStrategy
        if issubclass(cls, FixedThinkingStrategy):
            return cls(**kwargs)
        return cls(self.llm, self.evaluator, **kwargs)

    def from_config(self, config: "CoRTConfig", **kwargs) -> ThinkingStrategy:
        """Create a strategy from a :class:`CoRTConfig` object."""
        return self.create(config.thinking_strategy, **kwargs)


def load_strategy(
    name: str,
    llm: LLMProvider,
    evaluator: QualityEvaluator,
    **kwargs,
) -> ThinkingStrategy:
    """Load a thinking strategy by name with fallback to adaptive."""

    cls = get_strategy(name) or AdaptiveThinkingStrategy
    if issubclass(cls, FixedThinkingStrategy):
        return cls(**kwargs)
    return cls(llm, evaluator, **kwargs)


def strategy_from_config(
    config: "CoRTConfig",
    llm: LLMProvider,
    evaluator: QualityEvaluator,
    **kwargs,
) -> ThinkingStrategy:
    """Convenience wrapper to create a strategy from configuration."""
    return load_strategy(config.thinking_strategy, llm, evaluator, **kwargs)
