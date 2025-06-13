"""Thinking strategies and dynamic registry."""

from __future__ import annotations

from importlib.metadata import entry_points, EntryPoint
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
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            eps = eps.select(group=group)
        else:  # pragma: no cover - older importlib
            eps = eps.get(group, [])
    except Exception:  # pragma: no cover - best effort
        eps = []

    for ep in eps:
        try:
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, ThinkingStrategy):
                register_strategy(ep.name, obj)
        except Exception:  # pragma: no cover - best effort
            pass


register_strategy("adaptive", AdaptiveThinkingStrategy)
register_strategy("fixed", FixedThinkingStrategy)
register_strategy("hybrid", HybridToolStrategy)

_load_entrypoints()

from .factory import StrategyFactory, load_strategy, strategy_from_config  # noqa: E402


__all__ = [
    "ThinkingStrategy",
    "AdaptiveThinkingStrategy",
    "FixedThinkingStrategy",
    "HybridToolStrategy",
    "StrategyFactory",
    "load_strategy",
    "strategy_from_config",
    "register_strategy",
    "get_strategy",
    "available_strategies",
]
