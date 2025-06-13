import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.providers.resilient_llm import ResilientLLMProvider  # noqa: E402
from core.providers.llm import LLMProvider, StandardLLMResponse  # noqa: E402
from core.interfaces import QualityEvaluator  # noqa: E402
from core.strategies import (  # noqa: E402
    load_strategy,
    AdaptiveThinkingStrategy,
    FixedThinkingStrategy,
)
from core.budget import BudgetManager  # noqa: E402
from exceptions import APIError, TokenLimitError  # noqa: E402


class FailingLLM(LLMProvider):
    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        self.calls += 1
        raise self.exc


class SuccessLLM(LLMProvider):
    def __init__(self, content: str = "ok"):
        self.content = content
        self.calls = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        self.calls += 1
        return StandardLLMResponse(
            content=self.content,
            usage={"total_tokens": 2},
            model="success",
        )


class DummyEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_full_session_with_strategy_switch_and_budget():
    fail = FailingLLM(APIError("boom"))
    success = SuccessLLM()
    provider = ResilientLLMProvider(
        [fail, success], enable_hedging=False, max_retries=1
    )

    budget = BudgetManager("success", token_limit=5, catalog=[{"id": "success"}])
    tokenizer = SimpleNamespace(encode=lambda t: t.split())
    context = ContextManager(100, tokenizer)
    evaluator = DummyEvaluator()
    cache = InMemoryLRUCache(max_size=2)

    strategy = load_strategy("fixed", provider, evaluator, rounds=1)
    assert isinstance(strategy, FixedThinkingStrategy)

    engine = RecursiveThinkingEngine(
        llm=provider,
        cache=cache,
        evaluator=evaluator,
        context_manager=context,
        thinking_strategy=strategy,
        model_selector=None,
        budget_manager=budget,
    )

    async def _score_response(self, response: str, prompt: str) -> float:
        return evaluator.score(response, prompt)

    engine._score_response = _score_response.__get__(engine, RecursiveThinkingEngine)

    result = await engine.think_and_respond(
        "hi",
        thinking_rounds=1,
        alternatives_per_round=1,
    )
    assert result.response == "ok"
    assert fail.calls >= 1
    assert success.calls >= 1

    engine.thinking_strategy = load_strategy("unknown", provider, evaluator)
    assert isinstance(engine.thinking_strategy, AdaptiveThinkingStrategy)

    with pytest.raises(TokenLimitError):
        await engine.think_and_respond(
            "next",
            thinking_rounds=1,
            alternatives_per_round=1,
        )

    assert budget.tokens_used == 4
