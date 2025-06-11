import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import asyncio  # noqa: E402
from typing import List, Dict  # noqa: E402

from core.adaptive_thinking import AdaptiveThinkingAgent  # noqa: E402
from core.interfaces import LLMProvider, QualityEvaluator  # noqa: E402


class DummyLLM(LLMProvider):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.idx = 0

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.7) -> str:
        resp = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return resp


class DummyEval(QualityEvaluator):
    def __init__(self, scores):
        self.scores = scores

    def score(self, response: str, prompt: str) -> float:
        return self.scores.get(response, 0.0)


async def run_thinker():
    llm = DummyLLM(["bad", "better", "best", "best"])
    evaluator = DummyEval({"bad": 0.1, "better": 0.6, "best": 0.96})
    thinker = AdaptiveThinkingAgent(llm, evaluator)
    return await thinker.think("hi", max_rounds=5)


def test_adaptive_thinking_converges():
    result = asyncio.run(run_thinker())
    assert result == "best"
