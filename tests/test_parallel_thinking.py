import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import asyncio  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from core.optimization.parallel_thinking import (  # noqa: E402
    ParallelThinkingOptimizer,
    BatchThinkingOptimizer,
)
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402
from core.chat_v2 import CoRTConfig  # noqa: E402
from core.recursive_engine_v2 import create_optimized_engine  # noqa: E402
from core.model_router import ModelRouter  # noqa: E402
from core.budget import BudgetManager  # noqa: E402


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


def test_engine_parallel_flag():
    cfg = CoRTConfig(enable_parallel_thinking=False)
    router = ModelRouter.from_config(cfg)
    budget = BudgetManager(cfg.model, token_limit=cfg.budget_token_limit, catalog=[{"id": cfg.model, "pricing": {}}])
    engine = create_optimized_engine(cfg, router=router, budget_manager=budget)
    assert engine.parallel_optimizer is None

    cfg = CoRTConfig(enable_parallel_thinking=True)
    router = ModelRouter.from_config(cfg)
    budget = BudgetManager(cfg.model, token_limit=cfg.budget_token_limit, catalog=[{"id": cfg.model, "pricing": {}}])
    engine = create_optimized_engine(cfg, router=router, budget_manager=budget)
    assert engine.parallel_optimizer is not None


def test_threshold_propagation_parallel_engine():
    cfg = CoRTConfig(enable_parallel_thinking=True, quality_thresholds={"overall": 0.75})
    router = ModelRouter.from_config(cfg)
    budget = BudgetManager(cfg.model, token_limit=cfg.budget_token_limit, catalog=[{"id": cfg.model, "pricing": {}}])
    engine = create_optimized_engine(cfg, router=router, budget_manager=budget)
    assert engine.evaluator.thresholds["overall"] == 0.75
    assert engine.parallel_optimizer.quality_threshold == 0.75


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
