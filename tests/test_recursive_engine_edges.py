import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest  # noqa: E402
import json  # noqa: E402
from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402
from core.recursion import ConvergenceStrategy  # noqa: E402


class DummyLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = responses
        self.call = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        content = self.responses[self.call]
        self.call += 1
        return type(
            "Resp",
            (),
            {
                "content": content,
                "usage": {"total_tokens": 1},
                "model": "dummy",
                "cached": False,
            },
        )()


class DummyEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        return 0.5


class NoOpStrategy:
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(self, rounds_completed, quality_scores, responses):
        return rounds_completed < 1, "done"


class NumericEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        try:
            return float(response)
        except ValueError:
            return 0.0


class MultiRoundStrategy:
    def __init__(self, rounds: int = 3) -> None:
        self.rounds = rounds

    async def determine_rounds(self, prompt: str, *, request_id: str | None = None) -> int:
        return self.rounds

    async def should_continue(
        self,
        rounds_completed,
        quality_scores,
        responses,
        *,
        request_id: str | None = None,
    ):
        return True, "continue"


@pytest.mark.asyncio
async def test_invalid_json_alternative():
    llm = DummyLLM(["initial", "not json"])
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(
            100,
            type("T", (), {"encode": lambda self, t: t.split()})(),
        ),
        thinking_strategy=NoOpStrategy(),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            lambda r, p: 0.0,
            max_iterations=2,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.response == "not json"
    assert result.thinking_history[1].explanation.startswith("JSON parsing failed")


@pytest.mark.asyncio
async def test_selection_out_of_range():
    invalid = '{"alternatives": ["a"], "selection": "2", "thinking": "pick second"}'
    llm = DummyLLM(["initial", invalid])
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(
            100,
            type("T", (), {"encode": lambda self, t: t.split()})(),
        ),
        thinking_strategy=NoOpStrategy(),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            lambda r, p: 0.0,
            max_iterations=2,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.response == "initial"


@pytest.mark.asyncio
async def test_loop_respects_max_iterations():
    llm = DummyLLM(["initial", "ignored"])
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(
            100,
            type("T", (), {"encode": lambda self, t: t.split()})(),
        ),
        thinking_strategy=NoOpStrategy(),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            lambda r, p: 0.0,
            max_iterations=1,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.thinking_rounds == 0
    assert result.convergence_reason == "max iterations"


@pytest.mark.asyncio
async def test_loop_respects_time_limit(monkeypatch):
    llm = DummyLLM(["initial", "ignored"])
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(
            100,
            type("T", (), {"encode": lambda self, t: t.split()})(),
        ),
        thinking_strategy=NoOpStrategy(),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            lambda r, p: 0.0,
            max_iterations=5,
            time_limit=0.0,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.thinking_rounds == 0
    assert result.convergence_reason == "time limit"


@pytest.mark.asyncio
async def test_loop_quality_plateau():
    responses = [
        "0.1",
        json.dumps({"alternatives": ["0.11"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.111"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.1111"], "selection": "1", "thinking": ""}),
    ]
    llm = DummyLLM(responses)
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=NumericEvaluator(),
        context_manager=ContextManager(
            100, type("T", (), {"encode": lambda self, t: t.split()})()
        ),
        thinking_strategy=MultiRoundStrategy(rounds=3),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            NumericEvaluator().score,
            max_iterations=5,
            window=2,
            improvement_threshold=0.01,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.convergence_reason == "quality plateau"


@pytest.mark.asyncio
async def test_loop_statistical_convergence():
    seq = [
        "0.1",
        json.dumps({"alternatives": ["0.11"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.111"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.1112"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.1111"], "selection": "1", "thinking": ""}),
        json.dumps({"alternatives": ["0.11109"], "selection": "1", "thinking": ""}),
    ]
    llm = DummyLLM(seq)
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=NumericEvaluator(),
        context_manager=ContextManager(
            100, type("T", (), {"encode": lambda self, t: t.split()})()
        ),
        thinking_strategy=MultiRoundStrategy(rounds=5),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            NumericEvaluator().score,
            max_iterations=10,
            window=3,
            improvement_threshold=0.0005,
            advanced=True,
        ),
        model_selector=None,
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.convergence_reason in ["statistical convergence", "quality plateau"]
