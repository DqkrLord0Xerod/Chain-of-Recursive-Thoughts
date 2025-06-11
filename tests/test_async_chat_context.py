import pytest

from core.providers.llm import OpenRouterLLMProvider


class DummySession:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_context_closes_session(monkeypatch):
    monkeypatch.setattr('aiohttp.ClientSession', lambda *args, **kwargs: DummySession())

    provider = OpenRouterLLMProvider(api_key='k', model='m')
    async with provider as llm:
        assert llm._session is not None
    assert llm._session.closed
