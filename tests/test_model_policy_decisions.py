import importlib
import importlib.util
import pathlib
import types
import sys
from dataclasses import dataclass
import pytest

CORE_DIR = pathlib.Path(__file__).resolve().parents[1] / "core"

core_stub = types.ModuleType("core")
core_stub.interfaces = types.ModuleType("interfaces")
core_stub.interfaces.LLMProvider = object
sys.modules.setdefault("core", core_stub)
sys.modules.setdefault("core.interfaces", core_stub.interfaces)
exc_stub = types.ModuleType("exceptions")
exc_stub.APIError = Exception
sys.modules.setdefault("exceptions", exc_stub)
providers_stub = types.ModuleType("providers")


class DummyProvider:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")

    async def chat(self, messages, **kwargs):
        return None


providers_stub.OpenAILLMProvider = DummyProvider
providers_stub.OpenRouterLLMProvider = DummyProvider
providers_stub.MultiProviderLLM = type(
    "MultiProviderLLM",
    (),
    {"__init__": lambda self, providers: setattr(self, "providers", providers)},
)
providers_stub.LLMProvider = DummyProvider
core_stub.providers = providers_stub
sys.modules.setdefault("core.providers", providers_stub)

spec = importlib.util.spec_from_file_location(
    "model_policy", CORE_DIR / "model_policy.py"
)
model_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_policy)
sys.modules["core.model_policy"] = model_policy
ModelSelector = model_policy.ModelSelector

spec_router = importlib.util.spec_from_file_location(
    "model_router", CORE_DIR / "model_router.py"
)
model_router = importlib.util.module_from_spec(spec_router)
spec_router.loader.exec_module(model_router)
sys.modules["core.model_router"] = model_router
ModelRouter = model_router.ModelRouter


@dataclass
class DummyConfig:
    provider: str = "openai"
    api_key: str = "k"
    model: str = "m"
    providers: list[str] | None = None
    provider_weights: list[float] | None = None
    model_policy: dict | None = None
    max_retries: int = 3


def test_map_roles_with_default():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"assistant": "b", "default": "a"})
    result = selector.map_roles(["assistant", "critic"])
    assert result == {"assistant": "b", "critic": "a"}


@pytest.mark.asyncio
async def test_router_health(monkeypatch):
    async def fake_chat(messages, **kwargs):
        return type("Resp", (), {"content": "ok"})()

    metadata = [{"id": "a"}]
    selector = ModelSelector(metadata, {"assistant": "a"})
    cfg = DummyConfig()
    router = ModelRouter.from_config(cfg, selector)
    monkeypatch.setattr(
        "core.model_router.OpenAILLMProvider.chat",
        lambda self, m, **kw: fake_chat(m, **kw),
    )

    health = await router.provider_health()
    assert health == {"openai": True}
