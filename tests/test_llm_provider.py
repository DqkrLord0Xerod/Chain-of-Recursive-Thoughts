import pytest
import types

from core.providers.llm import (
    OpenRouterLLMProvider,
    OpenAILLMProvider,
    StandardLLMResponse,
)


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


@pytest.mark.asyncio
async def test_openai_llm_provider_chat(monkeypatch):
    class FakeChatCompletions:
        async def create(self, model, messages, temperature=0.7, max_tokens=None):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

        async def close(self):
            pass

    monkeypatch.setattr("core.providers.llm.openai.AsyncOpenAI", FakeClient)
    provider = OpenAILLMProvider(api_key="k", model="m")
    async with provider as llm:
        resp = await llm.chat([{"role": "user", "content": "hi"}])

    assert isinstance(resp, StandardLLMResponse)
    assert resp.content == "ok"
    assert resp.model == "m"
