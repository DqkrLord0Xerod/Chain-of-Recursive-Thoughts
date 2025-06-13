import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest  # noqa: E402

from core.tools import ToolRegistry, SearchTool, PythonExecutionTool  # noqa: E402
from core.strategies import HybridToolStrategy  # noqa: E402
from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.recursion import ConvergenceStrategy  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402


class DummyLLM(LLMProvider):
    def __init__(self):
        self.last_messages = None

    async def chat(self, messages, *, temperature=0.7, **kwargs):
        self.last_messages = messages
        return type("Resp", (), {"content": "done", "usage": {"total_tokens": 1}})()


class DummyEval(QualityEvaluator):
    thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_hybrid_strategy_invokes_tools():
    llm = DummyLLM()
    evaluator = DummyEval()
    registry = ToolRegistry()

    class DummySearch(SearchTool):
        async def run(self, task: str) -> str:
            return "cats info"

    class DummyPython(PythonExecutionTool):
        async def run(self, task: str) -> str:
            return "2"

    registry.register(DummySearch())
    registry.register(DummyPython())

    strategy = HybridToolStrategy(llm, evaluator, tools=registry)
    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=InMemoryLRUCache(max_size=2),
        evaluator=evaluator,
        context_manager=ContextManager(100, type("Tok", (), {"encode": lambda s, t: t.split()})()),
        tools=registry,
        thinking_strategy=strategy,
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            lambda r, p: 0.0,
            max_iterations=5,
        ),
    )

    await engine.think_and_respond("search: cats")
    assert "cats info" in llm.last_messages[-1]["content"]
