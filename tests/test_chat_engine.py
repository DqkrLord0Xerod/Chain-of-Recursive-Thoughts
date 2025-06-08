import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.chat_engine import ChatEngine
from core.cache_provider import InMemoryCache
from core.interfaces import LLMProvider
from core.quality_evaluator import DefaultQualityEvaluator


class DummyLLM(LLMProvider):
    async def chat(self, messages, *, temperature=0.7):
        prompt = messages[-1]["content"]
        return prompt.upper()


def test_engine_runs_event_loop():
    engine = ChatEngine(
        llm=DummyLLM(),
        cache=InMemoryCache(),
        evaluator=DefaultQualityEvaluator(lambda a, b: 1.0 if a == b.upper() else 0.0),
    )
    result = asyncio.run(engine.think("hello", max_rounds=1))
    assert result == "HELLO"
