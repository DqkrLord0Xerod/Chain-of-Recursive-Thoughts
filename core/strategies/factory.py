from __future__ import annotations

from core.interfaces import LLMProvider, QualityEvaluator

from .adaptive import AdaptiveThinkingStrategy
from .fixed import FixedThinkingStrategy
from .hybrid import HybridToolStrategy
from .base import ThinkingStrategy
from . import get_strategy


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
