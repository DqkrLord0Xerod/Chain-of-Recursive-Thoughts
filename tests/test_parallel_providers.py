import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "model_policy", os.path.join(ROOT, "core", "model_policy.py")
)
model_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_policy)

spec_i = importlib.util.spec_from_file_location(
    "interfaces", os.path.join(ROOT, "core", "interfaces.py")
)
interfaces = importlib.util.module_from_spec(spec_i)
spec_i.loader.exec_module(interfaces)

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

parallel_provider_call = model_policy.parallel_provider_call  # noqa: E402
LLMProvider = interfaces.LLMProvider  # noqa: E402


class DummyProvider(LLMProvider):
    def __init__(self, content, fail=False):
        self.content = content
        self.fail = fail

    async def chat(self, messages, *, temperature=0.7, max_tokens=None, metadata=None):
        if self.fail:
            raise RuntimeError("fail")
        await asyncio.sleep(0)
        return SimpleNamespace(content=self.content, usage={"total_tokens": 1}, model="m", cached=False)

    async def stream_chat(self, messages, *, temperature=0.7, max_tokens=None):
        yield self.content


@pytest.mark.asyncio
async def test_parallel_selects_best():
    p1 = DummyProvider("short")
    p2 = DummyProvider("much longer response")
    result = await parallel_provider_call([p1, p2], [
        {"role": "user", "content": "hi"}
    ], weights=[1.0, 0.5])
    assert result.content == "much longer response"


@pytest.mark.asyncio
async def test_parallel_handles_failure():
    p1 = DummyProvider("ok", fail=True)
    p2 = DummyProvider("fine")
    result = await parallel_provider_call([p1, p2], [
        {"role": "user", "content": "hi"}
    ])
    assert result.content == "fine"
