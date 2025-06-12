from __future__ import annotations

from core.interfaces import LLMProvider

from .adaptive import AdaptiveThinkingStrategy
from .fixed import FixedThinkingStrategy
from .base import ThinkingStrategy


_STRATEGY_MAP = {
    "adaptive": AdaptiveThinkingStrategy,
    "fixed": FixedThinkingStrategy,
}


def load_strategy(name: str, llm: LLMProvider, **kwargs) -> ThinkingStrategy:
    """Load a thinking strategy by name with fallback to adaptive."""
    cls = _STRATEGY_MAP.get(name.lower(), AdaptiveThinkingStrategy)
    if cls is FixedThinkingStrategy:
        return cls(**kwargs)
    return cls(llm, **kwargs)
