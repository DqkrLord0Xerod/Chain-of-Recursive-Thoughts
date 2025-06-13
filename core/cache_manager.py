"""Helper for caching LLM responses."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import structlog

from exceptions import TokenLimitError
from core.interfaces import (
    CacheProvider,
    LLMProvider,
    LLMResponse,
    EmbeddingProvider,
)
from config.config import CacheSettings
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
        embedding_provider: Optional[EmbeddingProvider] = None,
        cache_settings: Optional[CacheSettings] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.budget_manager = budget_manager
        self.model_selector = model_selector
        self.embedding_provider = embedding_provider

        settings = cache_settings or CacheSettings()
        self.semantic_enabled = settings.semantic_cache_enabled
        self.semantic_threshold = settings.semantic_cache_threshold
        self.semantic_max_entries = settings.semantic_cache_max_entries
        self.semantic_ttl = getattr(settings, "semantic_cache_ttl", 3600)
        self.semantic_min_hits = getattr(settings, "semantic_cache_min_hits", 0)
        self._semantic_entries: OrderedDict[str, Dict] = OrderedDict()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        role: str,
    ) -> LLMResponse:
        """Return cached response or call the LLM."""

        key = self._generate_key(messages, temperature)
        cached = await self.cache.get(key)
        if cached:
            logger.info("cache_hit", key=key[:8])
            if hasattr(cached, "cached"):
                cached.cached = True
            return cached

        # Check semantic cache
        semantic_hit = None
        if self.semantic_enabled and self.embedding_provider:
            query_text = json.dumps(messages, sort_keys=True)
            query_embedding = await self.embedding_provider.embed([query_text])
            if query_embedding:
                semantic_hit = await self._semantic_lookup(query_embedding[0])

        if semantic_hit is not None:
            resp = await self.cache.get(semantic_hit)
            if resp:
                logger.info("semantic_cache_hit", key=semantic_hit[:8])
                if hasattr(resp, "cached"):
                    resp.cached = True
                return resp

        if self.model_selector:
            self.llm.model = self.model_selector.model_for_role(role)

        response = await self.llm.chat(messages, temperature=temperature)

        if self.budget_manager:
            tokens = response.usage.get("total_tokens", 0)
            if self.budget_manager.will_exceed_budget(tokens):
                raise TokenLimitError("Token budget exceeded")
            self.budget_manager.record_usage(tokens)

        await self.cache.set(key, response, ttl=3600, tags=["llm_response"])

        if self.semantic_enabled and self.embedding_provider:
            query_text = json.dumps(messages, sort_keys=True)
            embedding = await self.embedding_provider.embed([query_text])
            if embedding:
                self._add_semantic_entry(key, query_text, embedding[0])

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

    async def _semantic_lookup(self, embedding: List[float]) -> Optional[str]:
        """Find best matching cached key for embedding."""
        self._prune_semantic_cache()

        best_key = None
        best_score = 0.0
        for key, entry in self._semantic_entries.items():
            score = self._cosine_similarity(embedding, entry["embedding"])
            if score >= self.semantic_threshold and score > best_score:
                best_key = entry["key"]
                best_score = score
                entry["hits"] += 1
                entry["accessed_at"] = time.time()

        return best_key

    def _add_semantic_entry(self, key: str, text: str, embedding: List[float]) -> None:
        """Add a new semantic cache entry."""
        self._semantic_entries[key] = {
            "key": key,
            "text": text,
            "embedding": embedding,
            "created_at": time.time(),
            "accessed_at": time.time(),
            "hits": 0,
        }
        self._prune_semantic_cache()

    def _prune_semantic_cache(self) -> None:
        """Remove stale or excess semantic cache entries."""
        now = time.time()
        for k in list(self._semantic_entries.keys()):
            entry = self._semantic_entries[k]
            if now - entry["accessed_at"] > self.semantic_ttl:
                del self._semantic_entries[k]

        if len(self._semantic_entries) <= self.semantic_max_entries:
            return

        sorted_items = sorted(
            self._semantic_entries.items(),
            key=lambda i: (i[1]["hits"], i[1]["accessed_at"]),
        )
        while len(self._semantic_entries) > self.semantic_max_entries:
            k, _ = sorted_items.pop(0)
            del self._semantic_entries[k]

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        from math import sqrt

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sqrt(sum(a * a for a in v1))
        norm2 = sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
