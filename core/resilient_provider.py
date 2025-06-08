from __future__ import annotations

import asyncio
from typing import List, Dict

from core.interfaces import LLMProvider
from exceptions import APIError, RateLimitError, TokenLimitError
from core.resilience import RetryState, with_retry


class ResilientLLMProvider(LLMProvider):
    """Wrap LLM providers with retry and fallback logic."""

    def __init__(self, primary: LLMProvider, fallbacks: List[LLMProvider] | None = None) -> None:
        self.primary = primary
        self.fallbacks = fallbacks or []

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        providers = [self.primary] + self.fallbacks
        for idx, provider in enumerate(providers):
            try:
                return await with_retry(
                    lambda: provider.chat(messages, temperature=temperature),
                    RetryState(),
                )
            except RateLimitError as e:
                if idx == len(providers) - 1:
                    raise e
                await asyncio.sleep(2 ** idx)
            except TokenLimitError as e:
                if idx == len(providers) - 1:
                    raise e
            except APIError:
                if idx == len(providers) - 1:
                    raise
        raise APIError("all providers failed")
