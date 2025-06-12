import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402
import types  # noqa: E402
mock_instr = types.ModuleType("opentelemetry.instrumentation.aiohttp_client")  # noqa: E402
mock_instr.AioHttpClientInstrumentor = type(
    "AioHttpClientInstrumentor",
    (),
    {"instrument": lambda *a, **k: None, "uninstrument": lambda *a, **k: None},
)
sys.modules.setdefault("opentelemetry.instrumentation.aiohttp_client", mock_instr)  # noqa: E402
mock_req = types.ModuleType("opentelemetry.instrumentation.requests")  # noqa: E402
mock_req.RequestsInstrumentor = type(
    "RequestsInstrumentor",
    (),
    {"instrument": lambda *a, **k: None, "uninstrument": lambda *a, **k: None},
)
sys.modules.setdefault("opentelemetry.instrumentation.requests", mock_req)  # noqa: E402

import pytest  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
from typing import Dict, List  # noqa: E402

from core.chat_v2 import RecursiveThinkingEngine  # noqa: E402
from core.context_manager import ContextManager  # noqa: E402
from core.conversation import ConversationManager  # noqa: E402
from core.recursion import ConvergenceStrategy  # noqa: E402
from core.providers.cache import InMemoryLRUCache  # noqa: E402
from core.memory import FaissMemoryStore  # noqa: E402


class DummyEmbeddingProvider:
    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] * 4 for t in texts]

    async def similarity(self, text1: str, text2: str) -> float:
        return 1.0


class MockLLMResponse:
    def __init__(self, content: str, tokens: int = 10):
        self.content = content
        self.usage = {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens}
        self.model = "test"
        self.cached = False


class MemoryAwareLLMProvider:
    def __init__(self):
        self.calls = []
        self.call_count = 0

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        **kwargs
    ) -> MockLLMResponse:
        self.calls.append({"messages": messages, "temperature": temperature})
        self.call_count += 1
        for msg in messages:
            if msg["role"] == "system":
                return MockLLMResponse(msg["content"])
        return MockLLMResponse("default")


class MockQualityEvaluator:
    def __init__(self):
        self.thresholds = {"overall": 0.5}

    def score(self, response: str, prompt: str) -> float:
        return 1.0


class MockThinkingStrategy:
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
    ) -> tuple[bool, str]:
        return False, "complete"


@pytest.mark.asyncio
async def test_memory_influences_response():
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: text.split()

    embedding_provider = DummyEmbeddingProvider()
    memory = FaissMemoryStore(embedding_provider, 4, top_k=1)
    await memory.add("The capital of France is Paris.")

    llm = MemoryAwareLLMProvider()
    cache = InMemoryLRUCache()
    evaluator = MockQualityEvaluator()
    context_manager = ContextManager(100, tokenizer)
    conversation = ConversationManager(llm, context_manager)
    strategy = MockThinkingStrategy()
    convergence = ConvergenceStrategy(lambda a, b: 1.0, evaluator.score)

    engine = RecursiveThinkingEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        context_manager=context_manager,
        thinking_strategy=strategy,
        convergence_strategy=convergence,
        model_selector=None,
        conversation_manager=conversation,
        memory_store=memory,
    )

    result = await engine.think_and_respond("What is the capital of France?")

    assert result.response == "The capital of France is Paris."
    first_messages = llm.calls[0]["messages"]
    assert any("Paris" in m["content"] for m in first_messages if m["role"] == "system")
