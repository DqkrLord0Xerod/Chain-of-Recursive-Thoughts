"""Helper for caching LLM responses."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

import structlog

from exceptions import TokenLimitError
from core.interfaces import CacheProvider, LLMProvider, LLMResponse
from core.model_policy import ModelSelector
from core.budget import BudgetManager

logger = structlog.get_logger(__name__)


class CacheManager:
    """Encapsulate caching logic for LLM calls."""

    def __init__(
        self,
        llm: LLMProvider,
        cache: CacheProvider,
        *,
        budget_manager: Optional[BudgetManager] = None,
        model_selector: Optional[ModelSelector] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.budget_manager = budget_manager
        self.model_selector = model_selector

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        role: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> LLMResponse:
        """Return cached response or call the LLM."""

        key = self._generate_key(messages, temperature)
        cached = await self.cache.get(key)
        if cached:
            logger.info(
                "cache_hit",
                key=key[:8],
                request_id=(metadata or {}).get("request_id"),
            )
            if hasattr(cached, "cached"):
                cached.cached = True
            return cached

        if self.model_selector:
            self.llm.model = self.model_selector.model_for_role(role)

        response = await self.llm.chat(
            messages,
            temperature=temperature,
            metadata=metadata,
        )

        if self.budget_manager:
            tokens = response.usage.get("total_tokens", 0)
            if self.budget_manager.will_exceed_budget(tokens):
                raise TokenLimitError("Token budget exceeded")
            self.budget_manager.record_usage(tokens)

        await self.cache.set(key, response, ttl=3600, tags=["llm_response"])
        return response

    def _generate_key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        content = json.dumps(
            {
                "messages": messages,
                "temperature": temperature,
                "model": getattr(self.llm, "model", "unknown"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()
