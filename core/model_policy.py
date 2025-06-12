from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import asyncio

from core.interfaces import LLMProvider
from exceptions import APIError


class ModelSelector:
    """Select models for roles based on a policy with fallbacks."""

    def __init__(self, metadata: Iterable[Dict[str, Any]], policy: Dict[str, str]):
        self._available = [m.get("id") for m in metadata if m.get("id")]
        self._available_set = set(self._available)
        if not self._available:
            raise ValueError("No model metadata provided")
        self.policy = policy

    def model_for_role(self, role: str) -> str:
        preferred = self.policy.get(role)
        if preferred and preferred in self._available_set:
            return preferred
        default = self.policy.get("default")
        if default and default in self._available_set:
            return default
        return self._available[0]

    def map_roles(self, roles: Iterable[str]) -> Dict[str, str]:
        return {role: self.model_for_role(role) for role in roles}


async def parallel_provider_call(
    providers: List[LLMProvider],
    messages: List[Dict[str, str]],
    *,
    weights: Optional[List[float]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Any:
    """Call providers concurrently and return the highest ranked response."""

    if not providers:
        raise ValueError("No providers supplied")

    weights = weights or [1.0] * len(providers)
    if len(weights) != len(providers):
        raise ValueError("weights length must match providers length")

    async def _call(p: LLMProvider):
        try:
            return await p.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - debug logging
            return exc

    results = await asyncio.gather(*[_call(p) for p in providers])

    scored: List[tuple[float, Any]] = []
    for weight, result in zip(weights, results):
        if isinstance(result, Exception):
            continue
        score = len(result.content) * weight
        scored.append((score, result))

    if not scored:
        raise APIError("All providers failed")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
