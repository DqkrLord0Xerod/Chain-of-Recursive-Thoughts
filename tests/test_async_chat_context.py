import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
import types  # noqa: E402
cb_mod = types.ModuleType('cb')
cb_mod.CircuitBreaker = type('CircuitBreaker', (), {})
cb_mod.CircuitOpenError = type('CircuitOpenError', (Exception,), {})
sys.modules.setdefault('core.resilience.circuit_breaker', cb_mod)

import core.chat as chat  # noqa: E402

AsyncEnhancedRecursiveThinkingChat = chat.AsyncEnhancedRecursiveThinkingChat
CoRTConfig = chat.CoRTConfig


class DummySession:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_context_closes_session(monkeypatch):
    monkeypatch.setattr('aiohttp.ClientSession', lambda: DummySession())

    chat = AsyncEnhancedRecursiveThinkingChat(CoRTConfig(api_key='k', model='m'))
    async with chat:
        assert chat.llm_client.session is not None
    assert chat.llm_client.session.closed
