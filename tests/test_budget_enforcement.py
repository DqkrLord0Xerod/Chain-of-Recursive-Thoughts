import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import pytest  # noqa: E402
from core.budget import BudgetManager  # noqa: E402
from exceptions import TokenLimitError  # noqa: E402
from core.cache_manager import CacheManager  # noqa: E402
from core.conversation import ConversationManager  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from tests.mocks import MockLLMProvider, MockCacheProvider  # noqa: E402


def test_budget_manager_loads_catalog(monkeypatch):
    calls = {}

    def fake_fetch():
        calls['called'] = True
        return [{
            "id": "m",
            "pricing": {"prompt": 0.001, "completion": 0.002}
        }]

    monkeypatch.setattr('core.budget.fetch_models', fake_fetch)
    manager = BudgetManager('m', token_limit=10)
    assert calls.get('called') is True

    manager.record_llm_usage(5)
    expected = 5 * (0.001 + 0.002) / 1000
    assert manager.dollars_spent == pytest.approx(expected)


def test_enforce_limit(monkeypatch):
    monkeypatch.setattr(
        BudgetManager,
        '_compute_cost_per_token',
        lambda self: 0.01,
    )
    manager = BudgetManager('x', token_limit=5, catalog=[{'id': 'x'}])

    assert not manager.will_exceed_budget(4)
    assert manager.will_exceed_budget(5)

    manager.enforce_limit(4)
    manager.record_llm_usage(4)
    assert manager.tokens_used == 4
    assert manager.dollars_spent == pytest.approx(0.04)
    with pytest.raises(TokenLimitError):
        manager.enforce_limit(2)


@pytest.mark.asyncio
async def test_cache_manager_logs_on_limit(monkeypatch):
    llm = MockLLMProvider(["too many tokens"])
    cache = MockCacheProvider()
    budget = BudgetManager("m", token_limit=2, catalog=[{"id": "m", "pricing": {}}])
    manager = CacheManager(llm, cache, budget_manager=budget)

    logs = []
    monkeypatch.setattr(
        "core.cache_manager.logger",
        type("L", (), {"info": lambda *a, **k: None})(),
    )
    monkeypatch.setattr(
        "core.budget.logger",
        type("L", (), {"warning": lambda *a, **k: logs.append(k)})(),
    )

    with pytest.raises(TokenLimitError):
        await manager.chat([{"role": "user", "content": "hi"}], temperature=0.7, role="assistant")

    assert logs


@pytest.mark.asyncio
async def test_conversation_manager_logs_on_limit(monkeypatch):
    llm = MockLLMProvider(["one two three"])
    tokenizer = type("Tok", (), {"encode": lambda self, t: t.split()})()
    context = ContextManager(100, tokenizer)
    budget = BudgetManager("m", token_limit=2, catalog=[{"id": "m", "pricing": {}}])
    convo = ConversationManager(llm, context, budget_manager=budget)
    convo.add("user", "hello")

    logs = []
    monkeypatch.setattr(
        "core.budget.logger",
        type("L", (), {"warning": lambda *a, **k: logs.append(k)})(),
    )

    with pytest.raises(TokenLimitError):
        await convo.summarize()

    assert logs
