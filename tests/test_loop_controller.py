import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import types
fake = types.ModuleType("core.chat_v2")

class ThinkingResult:
    pass

class ThinkingRound:
    pass

class CoRTConfig:
    pass

fake.ThinkingResult = ThinkingResult
fake.ThinkingRound = ThinkingRound
fake.CoRTConfig = CoRTConfig
sys.modules.setdefault("core.chat_v2", fake)
import pytest  # noqa: E402
from core.loop_controller import LoopController  # noqa: E402
from tests.mocks import MockLLMProvider, MockCacheProvider, MockQualityEvaluator  # noqa: E402


class DummyEngine:
    def __init__(self, cached_response: str | None = None) -> None:
        self.llm = MockLLMProvider(["initial"])
        self.cache = MockCacheProvider()
        self.evaluator = MockQualityEvaluator({"initial": 0.8})
        self.prompt_history: list[str] = []
        self.enable_compression = False
        self.enable_adaptive = False
        self.adaptive_optimizer = None
        self.parallel_optimizer = None
        self.cached_response = cached_response

    async def _check_semantic_cache(self, prompt: str):
        return self.cached_response

    async def _compress_prompt(self, prompt: str, context):
        return prompt

    async def _generate_initial(self, prompt: str, context):
        return await self.llm.chat([{"role": "user", "content": prompt}])

    async def _score_response(self, response: str, prompt: str) -> float:
        return self.evaluator.score(response, prompt)

    async def _update_semantic_cache(self, prompt: str, response: str, quality: float):
        self.updated = (prompt, response, quality)

    def _categorize_prompt(self, prompt: str) -> str:
        return "general"


@pytest.mark.asyncio
async def test_run_loop_cache_hit(monkeypatch):
    engine = DummyEngine(cached_response="cached")
    controller = LoopController(engine)
    monkeypatch.setattr("core.loop_controller.record_thinking_metrics", lambda *a, **k: None)

    result = await controller.run_loop("hello")

    assert result["cached"] is True
    assert result["response"] == "cached"


@pytest.mark.asyncio
async def test_run_loop_basic(monkeypatch):
    engine = DummyEngine()
    controller = LoopController(engine)
    monkeypatch.setattr("core.loop_controller.record_thinking_metrics", lambda *a, **k: None)

    result = await controller.run_loop("hello")

    assert result["cached"] is False
    assert result["response"] == "initial"
    assert engine.updated[0] == "hello"
    assert engine.prompt_history == ["hello"]
