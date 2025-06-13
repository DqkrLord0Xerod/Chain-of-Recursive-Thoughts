from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:  # pragma: no cover - import for typing
    from core.chat_v2 import CoRTConfig

from core.model_policy import ModelSelector
from core.budget import BudgetManager
from core.providers import (
    OpenAILLMProvider,
    OpenRouterLLMProvider,
    MultiProviderLLM,
    LLMProvider,
)


class ModelRouter:
    """Select providers and models for different roles."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        *,
        providers: Optional[List[str]] = None,
        provider_weights: Optional[List[float]] = None,
        model: Optional[str] = None,
        selector: Optional[ModelSelector] = None,
        max_retries: int = 3,
        budget_manager: Optional[BudgetManager] = None,
    ) -> None:
        self.provider = provider
        self.providers = providers or [provider]
        self.provider_weights = provider_weights
        self.api_key = api_key
        self.model = model
        self.selector = selector
        self.max_retries = max_retries
        self.budget_manager = budget_manager
        self.model_costs: Dict[str, float] = {}
        if budget_manager:
            for entry in budget_manager._load_catalog():
                mid = entry.get("id")
                pricing = entry.get("pricing", {})
                prompt = float(pricing.get("prompt", 0))
                completion = float(pricing.get("completion", 0))
                cost = (prompt + completion) / 1000.0 if prompt or completion else 0.0
                if mid:
                    self.model_costs[mid] = cost

    @classmethod
    def from_config(
        cls,
        config: "CoRTConfig",
        selector: Optional[ModelSelector] = None,
        budget_manager: Optional[BudgetManager] = None,
    ) -> "ModelRouter":
        return cls(
            provider=config.provider,
            api_key=config.api_key,
            providers=config.providers,
            provider_weights=config.provider_weights,
            model=config.model,
            selector=selector,
            max_retries=config.max_retries,
            budget_manager=budget_manager,
        )

    def model_for_role(self, role: str) -> str:
        model = self.selector.model_for_role(role) if self.selector else self.model or ""
        if self.budget_manager and self.model_costs:
            remaining = self.budget_manager.remaining_tokens
            threshold = 0.2 * self.budget_manager.token_limit
            if remaining <= threshold:
                cheapest = min(self.model_costs, key=self.model_costs.get)
                model = cheapest
        return model

    def _build_single_provider(self, name: str, model: str) -> LLMProvider:
        if name.lower() == "openai":
            return OpenAILLMProvider(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                model=model,
                max_retries=self.max_retries,
            )
        return OpenRouterLLMProvider(
            api_key=self.api_key or os.getenv("OPENROUTER_API_KEY"),
            model=model,
            max_retries=self.max_retries,
        )

    def provider_for_role(self, role: str) -> LLMProvider:
        model = self.model_for_role(role)
        health = {}
        try:
            health = asyncio.run(self.provider_health())
        except RuntimeError:
            # Fallback if running loop is active
            health = {}
        available = [p for p in self.providers if health.get(p, True)]
        if not available:
            available = list(self.providers)

        if len(available) == 1:
            return self._build_single_provider(available[0], model)

        if self.provider_weights and len(self.provider_weights) == len(self.providers):
            weight_map = dict(zip(self.providers, self.provider_weights))
            weights = [weight_map[p] for p in available]
            choice = random.choices(available, weights=weights, k=1)[0]
            return self._build_single_provider(choice, model)

        providers = [self._build_single_provider(p, model) for p in available]
        return MultiProviderLLM(providers)

    async def provider_health(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        ping = [{"role": "system", "content": "ping"}]
        for name in self.providers:
            provider = self._build_single_provider(name, self.model_for_role("assistant"))
            try:
                async with provider as p:
                    await p.chat(ping, temperature=0.0, max_tokens=1)
                results[name] = True
            except Exception:
                results[name] = False
        return results


__all__ = ["ModelRouter"]
