import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from core.interfaces import LLMProvider, QualityEvaluator
from core.providers.cache import InMemoryLRUCache
from core.recursive_engine_v2 import OptimizedRecursiveEngine
from monitoring.telemetry import initialize_telemetry


class DummyLLM(LLMProvider):
    def __init__(self):
        self.calls = []

    async def chat(self, messages, *, temperature: float = 0.7, **kwargs):
        self.calls.append(messages)
        return type(
            "Resp",
            (),
            {"content": "ok", "usage": {"total_tokens": 1}, "model": "dummy", "cached": False},
        )()


class DummyEval(QualityEvaluator):
    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_prompt_updates_across_rounds():
    llm = DummyLLM()
    initialize_telemetry(enable_prometheus=False)
    engine = OptimizedRecursiveEngine(
        llm=llm,
        cache=InMemoryLRUCache(),
        evaluator=DummyEval(),
    )

    await engine.think("first")
    assert llm.calls[0][-1]["content"] == "first"

    await engine.think("second")
    assert llm.calls[1][-1]["content"] == "first -> second"

