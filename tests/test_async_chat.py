import sys
import os
import asyncio
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import (  # noqa: E402
    AsyncEnhancedRecursiveThinkingChat,
    CoRTConfig,
)


class DummyResponse:
    def __init__(self, text):
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


class DummySession:
    def __init__(self, text):
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def post(self, *a, **k):
        return DummyResponse(self.text)


def test_async_call_api(monkeypatch):
    chat = AsyncEnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: DummySession("hi"))
    result = asyncio.run(chat._async_call_api([{"role": "user", "content": "hi"}]))
    assert result == "hi"


def test_parallel_generation(monkeypatch):
    chat = AsyncEnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))

    async def fake_api(messages, temperature=0.7, stream=False):
        idx = int(messages[-1]["content"].split("#")[1].split()[0])
        await asyncio.sleep(0)
        return f"alt{idx}"

    monkeypatch.setattr(chat, "_async_call_api", fake_api)
    alts = asyncio.run(
        chat._parallel_alternative_generation("base", "prompt", num_alternatives=3)
    )
    assert alts == ["alt1", "alt2", "alt3"]
