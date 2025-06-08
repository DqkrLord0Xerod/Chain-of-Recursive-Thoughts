import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import asyncio  # noqa: E402
from typing import List, Dict  # noqa: E402

from core.resilient_provider import ResilientLLMProvider  # noqa: E402
from core.interfaces import LLMProvider  # noqa: E402
from exceptions import APIError, RateLimitError  # noqa: E402


class FailingLLM(LLMProvider):
    def __init__(self, exc):
        self.exc = exc

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        raise self.exc


class SuccessLLM(LLMProvider):
    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        return "ok"


def test_resilient_provider_uses_fallback():
    provider = ResilientLLMProvider(
        FailingLLM(APIError("boom")), [SuccessLLM()]
    )
    result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    assert result == "ok"


def test_resilient_provider_reraises():
    provider = ResilientLLMProvider(FailingLLM(RateLimitError("limit")))
    try:
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
    except RateLimitError:
        assert True
    else:
        assert False
