import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest  # noqa: E402
from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402


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
    def score(self, response: str, prompt: str) -> float:
        return 0.5


class NoOpStrategy:
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(self, rounds_completed, quality_scores, responses):
        return rounds_completed < 1, "done"


@pytest.mark.asyncio
async def test_invalid_json_alternative():
    llm = DummyLLM(["initial", "not json"])
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(100, type("T", (), {"encode": lambda self, t: t.split()})()),
        thinking_strategy=NoOpStrategy(),
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
        context_manager=ContextManager(100, type("T", (), {"encode": lambda self, t: t.split()})()),
        thinking_strategy=NoOpStrategy(),
    )

    result = await engine.think_and_respond("prompt", alternatives_per_round=1)
    assert result.response == "initial"
