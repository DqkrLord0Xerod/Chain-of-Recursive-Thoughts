from typing import List, Dict
import pytest

from core.providers.resilient_llm import ResilientLLMProvider
from core.interfaces import LLMProvider, LLMResponse
from core.providers.llm import StandardLLMResponse
from exceptions import APIError, RateLimitError


class FailingLLM(LLMProvider):
    def __init__(self, exc: Exception):
        self.exc = exc

    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        raise self.exc


class SuccessLLM(LLMProvider):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        return StandardLLMResponse(
            content="ok",
            usage={"total_tokens": 1},
            model="success",
        )


@pytest.mark.asyncio
async def test_resilient_provider_uses_fallback():
    provider = ResilientLLMProvider(
        [FailingLLM(APIError("boom")), SuccessLLM()]
    )
    resp = await provider.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_resilient_provider_reraises():
    provider = ResilientLLMProvider([FailingLLM(RateLimitError("limit"))])
    with pytest.raises(RateLimitError):
        await provider.chat([{"role": "user", "content": "hi"}])
