from typing import Dict, List

import pytest

from core.chat_v2 import RecursiveThinkingEngine, ThinkingStrategy
from core.context_manager import ContextManager
from core.providers.cache import InMemoryLRUCache
from core.interfaces import LLMProvider, QualityEvaluator


class DummyLLM(LLMProvider):
    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7, **kwargs):
        return type('Resp', (), {
            'content': messages[-1]['content'].upper(),
            'usage': {'total_tokens': 1},
            'model': 'dummy',
            'cached': False,
        })()


class DummyEvaluator(QualityEvaluator):
    thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return 1.0 if response == prompt.upper() else 0.0


class OneRoundStrategy(ThinkingStrategy):
    async def determine_rounds(self, prompt: str) -> int:
        return 1

    async def should_continue(self, rounds_completed: int, quality_scores: List[float], responses: List[str]):
        return False, 'done'


@pytest.mark.asyncio
async def test_engine_runs_event_loop():
    tokenizer = type('Tok', (), {'encode': lambda self, t: t.split()})()
    engine = RecursiveThinkingEngine(
        llm=DummyLLM(),
        cache=InMemoryLRUCache(max_size=2),
        evaluator=DummyEvaluator(),
        context_manager=ContextManager(100, tokenizer),
        thinking_strategy=OneRoundStrategy(),
        model_selector=None,
    )
    result = await engine.think_and_respond('hello', thinking_rounds=1, alternatives_per_round=1)
    assert result.response == 'HELLO'
