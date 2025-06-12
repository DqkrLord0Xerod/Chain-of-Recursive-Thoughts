import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import json  # noqa: E402
from typing import Dict, List  # noqa: E402
import pytest  # noqa: E402

from core.planning import ImprovementPlanner  # noqa: E402
from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.recursion import ConvergenceStrategy  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402


class MockLLMProvider(LLMProvider):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call = 0

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7, **kwargs):
        content = self.responses[min(self.call, len(self.responses) - 1)]
        self.call += 1
        return type(
            "Resp",
            (),
            {"content": content, "usage": {"total_tokens": 1}, "model": "mock", "cached": False},
        )()


class DummyEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        return 0.5


class DummyStrategy:
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(self, rounds_completed: int, quality_scores: List[float], responses: List[str]):
        return rounds_completed < 1, "done"


@pytest.mark.asyncio
async def test_planner_output_actionable():
    llm = MockLLMProvider(["1. Clarify\n2. Add examples"])
    planner = ImprovementPlanner(llm)
    plan = await planner.create_plan("Prompt", "Resp")
    assert "1." in plan and "2." in plan


@pytest.mark.asyncio
async def test_engine_stores_plans():
    responses = [
        "initial",
        "1. Improve",
        json.dumps({"alternatives": [], "selection": "current", "thinking": "ok"}),
    ]
    llm = MockLLMProvider(responses)
    cache = InMemoryLRUCache()
    evaluator = DummyEvaluator()
    tokenizer = type("Tok", (), {"encode": lambda self, t: t.split()})()
    context_manager = ContextManager(100, tokenizer)
    strategy = DummyStrategy()
    convergence = ConvergenceStrategy(lambda a, b: 0.0, lambda r, p: 0.0)
    planner = ImprovementPlanner(llm)
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        context_manager=context_manager,
        thinking_strategy=strategy,
        convergence_strategy=convergence,
        model_selector=None,
        planner=planner,
    )
    result = await engine.think_and_respond("Prompt", thinking_rounds=1, alternatives_per_round=1)
    assert result.metadata["improvement_plans"][0].startswith("1.")
