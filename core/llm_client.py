from __future__ import annotations

import asyncio
import random
import time
from typing import List, Dict, Optional

import aiohttp

from api import openrouter
from exceptions import APIError, RateLimitError, TokenLimitError
from config import settings


class LLMClient:
    """Handle all communication with the OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.frontend_url,
            "X-Title": "Recursive Thinking Chat",
            "Content-Type": "application/json",
        }
        self.max_retries = max_retries
        self.failure_count = 0
        self.circuit_open = False
        self.circuit_reset_time = 0.0
        self.session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    # Synchronous call helper used by EnhancedRecursiveThinkingChat
    def chat(
        self,
        messages: List[Dict],
        *,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> str:
        if self.circuit_open:
            if time.time() < self.circuit_reset_time:
                raise APIError("Circuit breaker open")
            self.circuit_open = False
            self.failure_count = 0
        for attempt in range(1, self.max_retries + 1):
            try:
                return openrouter.sync_chat_completion(
                    self.headers,
                    messages,
                    self.model,
                    temperature=temperature,
                    stream=stream,
                )
            except (APIError, RateLimitError, TokenLimitError, Exception) as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    self.circuit_open = True
                    self.circuit_reset_time = time.time() + 2 ** attempt
                    raise e
                delay = 2 ** (attempt - 1) + random.uniform(0, 1)
                time.sleep(delay)
        raise APIError("Failed to get response")

    async def async_chat(
        self,
        messages: List[Dict],
        *,
        temperature: float = 0.7,
    ) -> str:
        if self.circuit_open:
            if time.time() < self.circuit_reset_time:
                raise APIError("Circuit breaker open")
            self.circuit_open = False
            self.failure_count = 0
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.session is None:
                    self.session = aiohttp.ClientSession()
                return await openrouter.async_chat_completion(
                    self.headers,
                    messages,
                    self.model,
                    temperature=temperature,
                    session=self.session,
                )
            except (APIError, RateLimitError, TokenLimitError, Exception) as e:
                self.failure_count += 1
                if attempt == self.max_retries:
                    self.circuit_open = True
                    self.circuit_reset_time = time.time() + 2 ** attempt
                    raise e
                delay = 2 ** (attempt - 1) + random.uniform(0, 1)
                await asyncio.sleep(delay)
        raise APIError("Failed to get response")

    def embeddings(self, texts: List[str]):
        return openrouter.get_embeddings(self.headers, texts)

