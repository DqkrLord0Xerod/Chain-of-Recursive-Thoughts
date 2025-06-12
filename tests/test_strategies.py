import pytest
from core.strategies import load_strategy, AdaptiveThinkingStrategy, FixedThinkingStrategy
from core.chat_v2 import CoRTConfig, create_default_engine


class DummyLLM:
    async def chat(self, *args, **kwargs):
        class Resp:
            content = "1"
        return Resp()


@pytest.mark.asyncio
async def test_load_strategy_known():
    llm = DummyLLM()
    strat = load_strategy("fixed", llm, rounds=2)
    assert isinstance(strat, FixedThinkingStrategy)
    rounds = await strat.determine_rounds("test")
    assert rounds == 2


@pytest.mark.asyncio
async def test_load_strategy_fallback():
    llm = DummyLLM()
    strat = load_strategy("unknown", llm)
    assert isinstance(strat, AdaptiveThinkingStrategy)


@pytest.mark.asyncio
async def test_engine_strategy_switch():
    cfg = CoRTConfig(thinking_strategy="fixed")
    engine = create_default_engine(cfg)
    assert isinstance(engine.thinking_strategy, FixedThinkingStrategy)


