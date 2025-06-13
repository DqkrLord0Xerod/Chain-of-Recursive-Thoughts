from __future__ import annotations

import os
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for typing
    from core.chat_v2 import CoRTConfig

from core.model_policy import ModelSelector
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
    ) -> None:
        self.provider = provider
        self.providers = providers or [provider]
        self.provider_weights = provider_weights
        self.api_key = api_key
        self.model = model
        self.selector = selector
        self.max_retries = max_retries

    @classmethod
    def from_config(
        cls, config: "CoRTConfig", selector: Optional[ModelSelector] = None
    ) -> "ModelRouter":
        return cls(
            provider=config.provider,
            api_key=config.api_key,
            providers=config.providers,
            provider_weights=config.provider_weights,
            model=config.model,
            selector=selector,
            max_retries=config.max_retries,
        )

    def model_for_role(self, role: str) -> str:
        if self.selector:
            return self.selector.model_for_role(role)
        return self.model or ""

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
        providers = [self._build_single_provider(p, model) for p in self.providers]
        if len(providers) == 1:
            return providers[0]
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
