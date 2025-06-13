import importlib.util
import importlib
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


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
OpenAILLMProvider = providers_stub.OpenAILLMProvider
MultiProviderLLM = providers_stub.MultiProviderLLM

budget_stub = types.ModuleType("budget")
budget_stub.BudgetManager = object
core_stub.budget = budget_stub
sys.modules.setdefault("core.budget", budget_stub)

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
core_stub.model_router = model_router
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


def test_selector_prefers_policy_model():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"assistant": "a"})
    assert selector.model_for_role("assistant") == "a"


def test_selector_falls_back_to_default():
    metadata = [{"id": "a"}, {"id": "b"}]
    selector = ModelSelector(metadata, {"critic": "c", "default": "b"})
    assert selector.model_for_role("critic") == "b"


def test_selector_first_available_when_missing():
    metadata = [{"id": "a"}]
    selector = ModelSelector(metadata, {"critic": "c"})
    assert selector.model_for_role("critic") == "a"


def test_selector_raises_for_no_models():
    with pytest.raises(ValueError):
        ModelSelector([], {"assistant": "a"})


def test_router_selects_provider_and_model():
    metadata = [{"id": "x"}]
    selector = ModelSelector(metadata, {"assistant": "x"})
    cfg = DummyConfig()
    router = ModelRouter.from_config(cfg, selector)
    provider = router.provider_for_role("assistant")
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.model == "x"


def test_router_multi_provider():
    cfg = DummyConfig(providers=["openai", "openrouter"])
    router = ModelRouter.from_config(cfg)
    provider = router.provider_for_role("assistant")
    assert isinstance(provider, MultiProviderLLM)
    assert len(provider.providers) == 2
