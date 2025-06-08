import sys
import os
import asyncio
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.chat import AsyncEnhancedRecursiveThinkingChat, CoRTConfig
from core.recursion import ConvergenceTracker

class DummyResponse:
    def __init__(self, text):
        self.text = text
        self.status = 200

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
    tracker = ConvergenceTracker(
        lambda a, b: 0.0,
        lambda r, p: 0.0,
        similarity_threshold=1.1,
        quality_threshold=-1.0,
    )
    async def gather():
        res = []
        async for alt in chat._parallel_alternative_generation(
            "base", "prompt", num_alternatives=3, tracker=tracker
        ):
            res.append(alt)
        return res

    alts = asyncio.run(gather())
    assert alts == ["alt1", "alt2", "alt3"]


def test_stream_generation_early_stop(monkeypatch):
    chat = AsyncEnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))

    responses = ["alt1", "alt2", "alt3"]

    async def fake_api(messages, temperature=0.7, stream=False):
        idx = int(messages[-1]["content"].split("#")[1].split()[0]) - 1
        await asyncio.sleep(0)
        return responses[idx]

    monkeypatch.setattr(chat, "_async_call_api", fake_api)
    monkeypatch.setattr(
        chat, "_semantic_similarity", lambda a, b: 0.96 if b == "alt2" else 0.1
    )
    scores = {"base": 0.5, "alt1": 0.6, "alt2": 0.61}
    monkeypatch.setattr(
        chat.quality_assessor,
        "comprehensive_score",
        lambda resp, prompt: {"overall": scores.get(resp, 0.0)},
    )

    async def gather_stop():
        res = []
        async for alt in chat._parallel_alternative_generation(
            "base", "prompt", num_alternatives=3
        ):
            res.append(alt)
        return res

    alts = asyncio.run(gather_stop())
    assert alts == ["alt1", "alt2"]
