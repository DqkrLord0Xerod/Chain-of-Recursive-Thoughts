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


def load_strategy(
    name: str,
    llm: LLMProvider,
    evaluator: QualityEvaluator,
    **kwargs,
) -> ThinkingStrategy:
    """Load a thinking strategy by name with fallback to adaptive."""
    cls = _STRATEGY_MAP.get(name.lower(), AdaptiveThinkingStrategy)
    if cls is FixedThinkingStrategy:
        return cls(**kwargs)
    return cls(llm, evaluator, **kwargs)
