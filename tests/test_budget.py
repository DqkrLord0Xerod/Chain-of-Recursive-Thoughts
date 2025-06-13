import pytest
from typing import Dict, List

from core.budget import BudgetManager
from core.chat_v2 import RecursiveThinkingEngine, ThinkingStrategy
from core.context_manager import ContextManager
from core.providers.cache import InMemoryLRUCache
from core.interfaces import LLMProvider, QualityEvaluator
from exceptions import TokenLimitError


class DummyLLM(LLMProvider):
    async def chat(
        self, messages: List[Dict[str, str]], *, temperature: float = 0.7, **kwargs
    ):
        return type(
            "Resp",
            (),
            {
                "content": messages[-1]["content"],
                "usage": {"total_tokens": 2},
                "model": "dummy",
                "cached": False,
            },
        )()


class DummyEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return 0.0


class TwoRoundStrategy(ThinkingStrategy):
    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        return 2

    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
        *,
        request_id: str,
    ):
        return rounds_completed < 2, "continue"


def test_budget_manager_records_usage():
    catalog = [{"id": "dummy", "pricing": {"prompt": 0.002, "completion": 0.003}}]
    manager = BudgetManager("dummy", token_limit=100, catalog=catalog)

    manager.record_llm_usage(50)
    expected_cost = 50 * (0.002 + 0.003) / 1000

    assert manager.tokens_used == 50
    assert manager.dollars_spent == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_engine_stops_on_budget():
    budget = BudgetManager("dummy", token_limit=3, catalog=[{"id": "dummy", "pricing": {}}])
    tokenizer = type("Tok", (), {"encode": lambda self, t: t.split()})()
    engine = RecursiveThinkingEngine(
        llm=DummyLLM(),
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(100, tokenizer),
        thinking_strategy=TwoRoundStrategy(),
        model_selector=None,
        budget_manager=budget,
    )

    with pytest.raises(TokenLimitError):
        await engine.think_and_respond("hello", thinking_rounds=2, alternatives_per_round=1)

    assert budget.tokens_used == 2
