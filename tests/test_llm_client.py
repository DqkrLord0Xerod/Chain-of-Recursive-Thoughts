import asyncio
import pytest
import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "llm_client", os.path.join(ROOT, "core", "llm_client.py")
)
llm_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_client)
LLMClient = llm_client.LLMClient


def test_sync_chat(monkeypatch):
    called = {}

    def fake_sync(headers, messages, model, temperature=0.7, stream=True):
        called['args'] = (headers, messages, model, temperature, stream)
        return "ok"

    monkeypatch.setattr("api.openrouter.sync_chat_completion", fake_sync)
    client = LLMClient(api_key="k", model="m", max_retries=1)
    resp = client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert resp == "ok"
    assert called['args'][2] == "m"


@pytest.mark.asyncio
async def test_async_chat(monkeypatch):
    async def fake_async(headers, messages, model, temperature=0.7, session=None):
        return "async"

    monkeypatch.setattr("api.openrouter.async_chat_completion", fake_async)
    client = LLMClient(api_key="k", model="m", max_retries=1)
    resp = await client.async_chat([{"role": "user", "content": "hi"}])
    await client.close()
    assert resp == "async"

