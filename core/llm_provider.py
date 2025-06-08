from __future__ import annotations

from typing import List, Dict

from api import openrouter
from config import settings
from core.interfaces import LLMProvider


class OpenRouterProvider(LLMProvider):
    """LLM provider using the OpenRouter API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.frontend_url,
            "X-Title": "Recursive Thinking Chat",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        return await openrouter.async_chat_completion(
            self.headers, messages, self.model, temperature=temperature
        )
