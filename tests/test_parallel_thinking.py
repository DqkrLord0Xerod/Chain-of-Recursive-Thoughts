import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import asyncio  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "parallel_thinking",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "optimization", "parallel_thinking.py"),
)
parallel_thinking = importlib.util.module_from_spec(spec)
sys.modules["parallel_thinking"] = parallel_thinking
spec.loader.exec_module(parallel_thinking)

ParallelThinkingOptimizer = parallel_thinking.ParallelThinkingOptimizer
BatchThinkingOptimizer = parallel_thinking.BatchThinkingOptimizer
spec_i = importlib.util.spec_from_file_location(
    "interfaces",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "core", "interfaces.py"),
)
interfaces = importlib.util.module_from_spec(spec_i)
sys.modules["interfaces"] = interfaces
spec_i.loader.exec_module(interfaces)
LLMProvider = interfaces.LLMProvider  # noqa: E402
QualityEvaluator = interfaces.QualityEvaluator  # noqa: E402


class DummyLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.1)
        return SimpleNamespace(content="alt", usage={"total_tokens": 1})


class DummyEval(QualityEvaluator):
    thresholds = {"overall": 1.0}

    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_parallel_generation():
    llm = DummyLLM()
    optimizer = ParallelThinkingOptimizer(
        llm,
        DummyEval(),
        max_parallel=3,
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


class DummyCritic:
    def __init__(self, scores):
        self.scores = scores

    async def score(self, response: str, prompt: str) -> float:
        return self.scores.get(response, 0.0)


class SeqLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        resp = self.responses[self.idx]
        self.idx += 1
        return SimpleNamespace(content=resp, usage={"total_tokens": 1})


class DelayLLM(LLMProvider):
    def __init__(self, responses, delays):
        self.responses = responses
        self.delays = delays
        self.idx = 0

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        i = self.idx
        self.idx += 1
        await asyncio.sleep(self.delays[i])
        return SimpleNamespace(content=self.responses[i], usage={"total_tokens": 1})


@pytest.mark.asyncio
async def test_critic_changes_selection():
    llm = SeqLLM(["a1", "a2"])
    critic = DummyCritic({"a1": 0.2, "a2": 0.8})
    opt = ParallelThinkingOptimizer(
        llm,
        DummyEval(),
        critic=critic,
        max_parallel=2,
        timeout_per_round=1.0,
    )

    best, candidates, _ = await opt.think_parallel(
        "p",
        "init",
        max_rounds=1,
        alternatives_per_round=2,
    )

    assert best == "a2"


@pytest.mark.asyncio
async def test_batch_optimizer_multiple_prompts():
    llm = DummyLLM()
    opt = ParallelThinkingOptimizer(
        llm,
        DummyEval(),
        max_parallel=3,
        timeout_per_round=1.0,
    )
    batch = BatchThinkingOptimizer(opt, batch_size=3, batch_timeout=0.01)

    start = time.perf_counter()
    results = await batch.think_batch(["p1", "p2", "p3"])
    duration = time.perf_counter() - start

    assert len(results) == 3
    assert all(r[0] == "alt" for r in results)
    assert duration < 0.5


@pytest.mark.asyncio
async def test_early_stop_speedup():
    delays = [0.1, 0.5, 0.5]
    responses = ["alt_1", "alt_2", "alt_3"]
    eval_ = DummyEval()
    eval_.score = lambda r, p: 0.9 if r == "alt_1" else 0.2

    opt_no_stop = ParallelThinkingOptimizer(
        DelayLLM(responses, delays),
        eval_,
        max_parallel=3,
        quality_threshold=1.0,
        timeout_per_round=1.0,
    )
    await opt_no_stop.think_parallel("p", "init", max_rounds=1, alternatives_per_round=3)

    opt_stop = ParallelThinkingOptimizer(
        DelayLLM(responses, delays),
        eval_,
        max_parallel=3,
        quality_threshold=0.8,
        timeout_per_round=1.0,
    )
    _, _, metrics = await opt_stop.think_parallel(
        "p",
        "init",
        max_rounds=1,
        alternatives_per_round=3,
    )

    assert metrics["early_stopped"] is True
    assert metrics["speedup"] > 0
