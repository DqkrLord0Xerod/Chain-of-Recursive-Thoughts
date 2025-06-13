import os
import sys
from dataclasses import dataclass
import importlib.util
import types
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import pytest  # noqa: E402

CORE_DIR = os.path.dirname(os.path.dirname(__file__)) + "/core"

core_stub = types.ModuleType("core")
core_stub.interfaces = types.ModuleType("interfaces")
core_stub.interfaces.LLMProvider = object
sys.modules.setdefault("core.interfaces", core_stub.interfaces)
sys.modules.setdefault("core", core_stub)

providers_stub = types.ModuleType("providers")


class _Provider:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")

    async def chat(self, messages, **kwargs):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


providers_stub.OpenAILLMProvider = _Provider
providers_stub.OpenRouterLLMProvider = _Provider
providers_stub.MultiProviderLLM = type(
    "MultiProviderLLM",
    (),
    {"__init__": lambda self, providers: setattr(self, "providers", providers)},
)
providers_stub.LLMProvider = _Provider
core_stub.providers = providers_stub
sys.modules.setdefault("core.providers", providers_stub)
exc_stub = types.ModuleType("exceptions")
exc_stub.APIError = Exception
sys.modules.setdefault("exceptions", exc_stub)

spec_pol = importlib.util.spec_from_file_location("model_policy", os.path.join(CORE_DIR, "model_policy.py"))
model_policy = importlib.util.module_from_spec(spec_pol)
spec_pol.loader.exec_module(model_policy)
sys.modules["core.model_policy"] = model_policy
ModelSelector = model_policy.ModelSelector


class BudgetManager:
    def __init__(self, model: str, token_limit: int, catalog=None) -> None:
        self.model = model
        self.token_limit = token_limit
        self.tokens_used = 0
        self._cost_per_token = 0.0
        for entry in catalog or []:
            if entry.get("id") == model:
                pricing = entry.get("pricing", {})
                p = float(pricing.get("prompt", 0))
                c = float(pricing.get("completion", 0))
                if p or c:
                    self._cost_per_token = (p + c) / 1000.0
        self._catalog = catalog or []

    def _load_catalog(self):
        return self._catalog

    @property
    def remaining_tokens(self) -> int:
        return self.token_limit - self.tokens_used

    @property
    def cost_per_token(self) -> float:
        return self._cost_per_token


sys.modules["core.budget"] = sys.modules[__name__]

spec_router = importlib.util.spec_from_file_location("model_router", os.path.join(CORE_DIR, "model_router.py"))
model_router = importlib.util.module_from_spec(spec_router)
spec_router.loader.exec_module(model_router)
sys.modules["core.model_router"] = model_router
core_stub.model_router = model_router
ModelRouter = model_router.ModelRouter


@dataclass
class DummyConfig:
    provider: str = "openai"
    api_key: str = "k"
    model: str = "exp"
    providers: list[str] | None = None
    provider_weights: list[float] | None = None
    model_policy: dict | None = None
    max_retries: int = 3


@pytest.mark.asyncio
async def test_provider_fallback_and_budget(monkeypatch):
    class BaseProvider:
        def __init__(self, *_, **kw):
            self.model = kw.get("model")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FailProvider(BaseProvider):
        async def chat(self, messages, **kwargs):
            raise Exception("boom")

    class SuccessProvider(BaseProvider):
        async def chat(self, messages, **kwargs):
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr("core.model_router.OpenAILLMProvider", FailProvider)
    monkeypatch.setattr("core.model_router.OpenRouterLLMProvider", SuccessProvider)

    metadata = [
        {"id": "exp", "pricing": {"prompt": 0.02, "completion": 0.02}},
        {"id": "cheap", "pricing": {"prompt": 0.001, "completion": 0.001}},
    ]
    selector = ModelSelector(metadata, {"assistant": "exp"})
    budget = BudgetManager("exp", token_limit=10, catalog=metadata)
    budget.tokens_used = 9
    cfg = DummyConfig(providers=["openai", "openrouter"])
    router = ModelRouter.from_config(cfg, selector, budget_manager=budget)

    provider = await asyncio.to_thread(router.provider_for_role, "assistant")

    assert isinstance(provider, SuccessProvider)
    assert provider.model == "cheap"
