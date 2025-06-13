import pytest
from core.strategies import (
    load_strategy,
    AdaptiveThinkingStrategy,
    FixedThinkingStrategy,
)
from core.strategies.base import ThinkingStrategy
from mypy import api
from core.chat_v2 import CoRTConfig, create_default_engine


class DummyLLM:
    async def chat(self, *args, **kwargs):
        class Resp:
            content = "1"
        return Resp()


class DummyEvaluator:
    thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_load_strategy_known():
    llm = DummyLLM()
    strat = load_strategy("fixed", llm, DummyEvaluator(), rounds=2)
    assert isinstance(strat, FixedThinkingStrategy)
    rounds = await strat.determine_rounds("test")
    assert rounds == 2


@pytest.mark.asyncio
async def test_load_strategy_fallback():
    llm = DummyLLM()
    strat = load_strategy("unknown", llm, DummyEvaluator())
    assert isinstance(strat, AdaptiveThinkingStrategy)


@pytest.mark.asyncio
async def test_engine_strategy_switch():
    cfg = CoRTConfig(thinking_strategy="fixed")
    engine = create_default_engine(cfg)
    assert isinstance(engine.thinking_strategy, FixedThinkingStrategy)


def test_threshold_propagation_default_engine():
    cfg = CoRTConfig(quality_thresholds={"overall": 0.8})
    engine = create_default_engine(cfg)
    assert engine.evaluator.thresholds["overall"] == 0.8
    assert engine.thinking_strategy.quality_threshold == 0.8


def test_strategy_isinstance():
    assert issubclass(AdaptiveThinkingStrategy, ThinkingStrategy)
    assert issubclass(FixedThinkingStrategy, ThinkingStrategy)


def test_mypy_check(tmp_path):
    code = """
from core.strategies.base import ThinkingStrategy
from typing import List


class Dummy(ThinkingStrategy):
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(self, rounds_completed: int, quality_scores: List[float], responses: List[str]) -> tuple[bool, str]:
        return False, 'done'
"""
    path = tmp_path / "snippet.py"
    path.write_text(code)
    result = api.run([str(path)])
    assert result[2] == 0, result[0]
