import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from importlib.metadata import EntryPoint

import pytest


pkg = ModuleType("core")
pkg.__path__ = [os.path.abspath("core")]
sys.modules["core"] = pkg

dummy = ModuleType("core.chat_v2")
dummy.ThinkingResult = object
dummy.ThinkingRound = object
sys.modules["core.chat_v2"] = dummy

spec = importlib.util.spec_from_file_location(
    "core.strategies", os.path.join("core", "strategies", "__init__.py")
)
strategies = importlib.util.module_from_spec(spec)
sys.modules["core.strategies"] = strategies
spec.loader.exec_module(strategies)


class DummyLLM:
    async def chat(self, *a, **k):
        return SimpleNamespace(content="1")


class DummyEval:
    thresholds = {"overall": 0.9}

    def score(self, resp: str, prompt: str) -> float:
        return 0.0


def fake_entry_points(*, group=None):
    if group == "mils_strategies":
        path = os.path.join(os.path.dirname(__file__), "mocks")
        sys.path.insert(0, path)
        return [EntryPoint("dummy", "strategy_plugin:PluginStrategy", group)]
    return []


@pytest.mark.asyncio
async def test_plugin_registry(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    importlib.reload(strategies)
    assert "dummy" in strategies.available_strategies()
    strat = strategies.load_strategy("dummy", DummyLLM(), DummyEval())
    assert strat.__class__.__name__ == "PluginStrategy"
