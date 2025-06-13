import os
import sys
from unittest.mock import MagicMock
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

mock_instr = types.ModuleType("opentelemetry.instrumentation.aiohttp_client")
mock_instr.AioHttpClientInstrumentor = type(
    "AioHttpClientInstrumentor",
    (),
    {"instrument": lambda *a, **k: None, "uninstrument": lambda *a, **k: None},
)
sys.modules.setdefault("opentelemetry.instrumentation.aiohttp_client", mock_instr)
mock_req = types.ModuleType("opentelemetry.instrumentation.requests")
mock_req.RequestsInstrumentor = type(
    "RequestsInstrumentor",
    (),
    {"instrument": lambda *a, **k: None, "uninstrument": lambda *a, **k: None},
)
sys.modules.setdefault("opentelemetry.instrumentation.requests", mock_req)

import pytest  # noqa: E402

from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.recursion import ConvergenceStrategy  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.loop_controller import LoopState  # noqa: E402


class DummyLLM:
    async def chat(self, messages, *, temperature=0.7, **kwargs):
        return type(
            "Resp",
            (),
            {
                "content": "reply",
                "usage": {"total_tokens": 1},
                "model": "dummy",
                "cached": False,
            },
        )()


class DummyEvaluator:
    thresholds = {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        return 0.6


class SimpleStrategy:
    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        return 1

    async def should_continue(
        self,
        rounds_completed,
        quality_scores,
        responses,
        *,
        request_id: str,
    ):
        return False, "complete"


@pytest.mark.asyncio
async def test_loop_state_persistence(tmp_path, monkeypatch):
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: text.split()
    context_manager = ContextManager(50, tokenizer)

    engine = RecursiveThinkingEngine(
        llm=DummyLLM(),
        cache=InMemoryLRUCache(),
        evaluator=DummyEvaluator(),
        context_manager=context_manager,
        thinking_strategy=SimpleStrategy(),
        convergence_strategy=ConvergenceStrategy(
            lambda a, b: 0.0,
            DummyEvaluator().score,
        ),
        model_selector=None,
    )

    async def _score(response, prompt):
        return engine.evaluator.score(response, prompt)

    engine._score_response = _score

    monkeypatch.setattr("core.loop_controller.SESSION_DIR", str(tmp_path))

    await engine.think_and_respond("Hi", session_id="sess1")

    history = await engine.loop_controller.load_loop_history("sess1")
    assert len(history) == 1
    assert isinstance(history[0], LoopState)
    reasons = await engine.loop_controller.get_convergence_reasons("sess1")
    assert reasons == ["complete"]
