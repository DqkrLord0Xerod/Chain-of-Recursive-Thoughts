import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import (  # noqa: E402
    EnhancedRecursiveThinkingChat,
    CoRTConfig,
)


def test_batch_generate_and_evaluate_single_call(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    calls = []

    def fake_call(messages, temperature=0.7, stream=True):
        calls.append(1)
        return json.dumps({
            "alternatives": ["a1", "a2"],
            "current": {
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "relevance": 5
            },
            "1": {
                "accuracy": 8,
                "completeness": 8,
                "clarity": 8,
                "relevance": 8
            },
            "choice": "1",
            "reason": "better"
        })

    monkeypatch.setattr(chat, "_call_api", fake_call)
    monkeypatch.setattr(
        chat.quality_assessor,
        "comprehensive_score",
        lambda *a, **k: {
            "relevance": 0.5,
            "completeness": 0.5,
            "clarity": 0.5,
            "accuracy": 0.5,
            "overall": 0.5,
        },
    )

    best, alts, reason = chat._batch_generate_and_evaluate(
        "base", "prompt", num_alternatives=2
    )

    assert len(calls) == 1
    assert alts == ["a1", "a2"]
    assert best == "a1"
    assert reason == "better"
