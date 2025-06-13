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

    def enforce_limit(self, next_tokens: int) -> None:
        """Raise ``TokenLimitError`` if the token budget would be exceeded."""
        from exceptions import TokenLimitError

        if self.will_exceed_budget(next_tokens):
            raise TokenLimitError("Token budget exceeded")

    def record_llm_usage(self, tokens: int) -> None:
        """Record ``tokens`` consumed and update cost statistics."""
        self.tokens_used += tokens
        self.dollars_spent += tokens * self._cost_per_token

    # Backwards compatibility
    record_usage = record_llm_usage

    @property
    def cost_per_token(self) -> float:
        """Return cost per token for the configured model."""
        return self._cost_per_token

    @property
    def remaining_tokens(self) -> int:
        """Return the number of tokens remaining in the budget."""
        return max(self.token_limit - self.tokens_used, 0)
