import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import pytest  # noqa: E402
from core.budget import BudgetManager  # noqa: E402
from exceptions import TokenLimitError  # noqa: E402
from core.cache_manager import CacheManager  # noqa: E402
from tests.mocks import MockLLMProvider, MockCacheProvider  # noqa: E402


@pytest.mark.asyncio
async def test_budget_manager_limits(monkeypatch):
    llm = MockLLMProvider(["too many tokens"])
    cache = MockCacheProvider()
    budget = BudgetManager("m", token_limit=2, catalog=[{"id": "m", "pricing": {}}])
    manager = CacheManager(llm, cache, budget_manager=budget)

    monkeypatch.setattr("core.cache_manager.logger", type("L", (), {"info": lambda *a, **k: None})())

    with pytest.raises(TokenLimitError):
        await manager.chat([{"role": "user", "content": "hi"}], temperature=0.7, role="assistant")

    assert budget.tokens_used == 0


@pytest.mark.asyncio
async def test_budget_manager_records(monkeypatch):
    llm = MockLLMProvider(["ok"])
    cache = MockCacheProvider()
    budget = BudgetManager("m", token_limit=10, catalog=[{"id": "m", "pricing": {}}])
    manager = CacheManager(llm, cache, budget_manager=budget)

    monkeypatch.setattr("core.cache_manager.logger", type("L", (), {"info": lambda *a, **k: None})())

    resp = await manager.chat([{"role": "user", "content": "hi"}], temperature=0.7, role="assistant")

    assert budget.tokens_used == resp.usage["total_tokens"]
    assert budget.dollars_spent == 0.0
