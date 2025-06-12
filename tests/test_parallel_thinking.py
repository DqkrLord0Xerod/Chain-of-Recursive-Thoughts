import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import asyncio  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from core.optimization.parallel_thinking import ParallelThinkingOptimizer  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402
from core.chat_v2 import CoRTConfig  # noqa: E402
from core.recursive_engine_v2 import create_optimized_engine  # noqa: E402


class DummyLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.1)
        return SimpleNamespace(content="alt", usage={"total_tokens": 1})


class DummyEval(QualityEvaluator):
    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_parallel_generation():
    llm = DummyLLM()
    optimizer = ParallelThinkingOptimizer(
        llm,
        DummyEval(),
        max_parallel=3,
        quality_threshold=1.0,
        timeout_per_round=1.0,
    )

    start = time.perf_counter()
    _, candidates, _ = await optimizer.think_parallel(
        "p",
        "init",
        max_rounds=1,
        alternatives_per_round=3,
    )
    duration = time.perf_counter() - start

    assert len(candidates) == 4
    assert duration < 0.25


def test_engine_parallel_flag():
    cfg = CoRTConfig(enable_parallel_thinking=False)
    engine = create_optimized_engine(cfg)
    assert engine.parallel_optimizer is None

    cfg = CoRTConfig(enable_parallel_thinking=True)
    engine = create_optimized_engine(cfg)
    assert engine.parallel_optimizer is not None
