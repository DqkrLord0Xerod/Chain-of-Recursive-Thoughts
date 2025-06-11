from __future__ import annotations

from typing import Any, Dict, Iterable


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
