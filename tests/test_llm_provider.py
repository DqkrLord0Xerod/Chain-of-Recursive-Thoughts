import pytest

from core.providers.llm import OpenRouterLLMProvider, StandardLLMResponse


@pytest.mark.asyncio
async def test_llm_provider_chat(monkeypatch):
    async def fake_async(headers, messages, model, temperature=0.7, session=None):
        return "ok"

    monkeypatch.setattr("api.openrouter.async_chat_completion", fake_async)
    provider = OpenRouterLLMProvider(api_key="k", model="m")
    async with provider as llm:
        resp = await llm.chat([{"role": "user", "content": "hi"}])

    assert isinstance(resp, StandardLLMResponse)
    assert resp.content == "ok"
    assert resp.model == "m"
