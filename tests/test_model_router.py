import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import pytest  # noqa: E402
from core.cache_manager import CacheManager  # noqa: E402
from core.model_policy import ModelSelector  # noqa: E402
from tests.mocks import MockLLMProvider, MockCacheProvider  # noqa: E402


@pytest.mark.asyncio
async def test_cache_manager_routes_and_caches(monkeypatch):
    llm = MockLLMProvider(["first"])
    cache = MockCacheProvider()
    selector = ModelSelector([{"id": "a"}, {"id": "b"}], {"assistant": "b"})
    manager = CacheManager(llm, cache, model_selector=selector)

    monkeypatch.setattr("core.cache_manager.logger", type("L", (), {"info": lambda *a, **k: None})())

    messages = [{"role": "user", "content": "hi"}]
    resp1 = await manager.chat(messages, temperature=0.7, role="assistant")
    assert llm.model == "b"
    resp2 = await manager.chat(messages, temperature=0.7, role="assistant")
    assert resp2.cached
    assert resp1.content == resp2.content
