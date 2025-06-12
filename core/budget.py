from __future__ import annotations

from typing import Dict, List, Optional

from api import fetch_models


class BudgetManager:
    """Track token usage and compute costs for a model."""

    def __init__(self, model: str, token_limit: int, catalog: Optional[List[Dict[str, any]]] = None) -> None:
        self.model = model
        self.token_limit = token_limit
        self.tokens_used = 0
        self.dollars_spent = 0.0

        catalog = catalog or self._load_catalog()
        self.pricing = self._find_pricing(catalog)
        self._cost_per_token = self._compute_cost_per_token()

    @staticmethod
    def _load_catalog() -> List[Dict[str, any]]:
        try:
            return fetch_models()
        except Exception:
            return []

    def _find_pricing(self, catalog: List[Dict[str, any]]) -> Dict[str, float]:
        for entry in catalog:
            if entry.get("id") == self.model:
                return entry.get("pricing", {})
        return {}

    def _compute_cost_per_token(self) -> float:
        prompt = float(self.pricing.get("prompt", 0))
        completion = float(self.pricing.get("completion", 0))
        if prompt == 0 and completion == 0:
            return 0.0
        return (prompt + completion) / 1000.0

    def will_exceed_budget(self, next_tokens: int) -> bool:
        """Return True if adding ``next_tokens`` would exceed the limit."""
        return self.tokens_used + next_tokens >= self.token_limit

    def record_usage(self, tokens: int) -> None:
        """Record ``tokens`` consumed and update cost statistics."""
        self.tokens_used += tokens
        self.dollars_spent += tokens * self._cost_per_token
