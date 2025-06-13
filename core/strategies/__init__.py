"""Thinking strategies and dynamic registry."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Dict, Type

from .base import ThinkingStrategy
from .adaptive import AdaptiveThinkingStrategy
from .hybrid import HybridToolStrategy
from .fixed import FixedThinkingStrategy

_REGISTRY: Dict[str, Type[ThinkingStrategy]] = {}


def register_strategy(name: str, cls: Type[ThinkingStrategy]) -> None:
    """Register a strategy implementation."""
    _REGISTRY[name.lower()] = cls


def get_strategy(name: str) -> Type[ThinkingStrategy] | None:
    """Retrieve a strategy class by name."""
    return _REGISTRY.get(name.lower())


def available_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return sorted(_REGISTRY)


def _load_entrypoints(group: str = "mils_strategies") -> None:
    """Load strategy plugins from entry points."""
    for ep in entry_points(group=group):
        try:
            cls = ep.load()
            register_strategy(ep.name, cls)
        except Exception:  # pragma: no cover - best effort
            pass


register_strategy("adaptive", AdaptiveThinkingStrategy)
register_strategy("fixed", FixedThinkingStrategy)
register_strategy("hybrid", HybridToolStrategy)

_load_entrypoints()

from .factory import load_strategy  # noqa: E402

__all__ = [
    "ThinkingStrategy",
    "AdaptiveThinkingStrategy",
    "FixedThinkingStrategy",
    "HybridToolStrategy",
    "load_strategy",
    "register_strategy",
    "get_strategy",
    "available_strategies",
]
