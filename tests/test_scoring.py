import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.chat import EnhancedRecursiveThinkingChat, CoRTConfig


def test_score_called(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    calls = []

    def fake_score(resp, prompt):
        calls.append(resp)
        return {
            "relevance": 0.5,
            "completeness": 0.5,
            "clarity": 0.5,
            "accuracy": 0.5,
            "overall": 0.5,
        }

    def fake_call(messages, temperature=0.7, stream=True):
        return json.dumps({"alternatives": ["b", "c"], "choice": "1"})

    monkeypatch.setattr(chat.quality_assessor, "comprehensive_score", fake_score)
    monkeypatch.setattr(chat, "_call_api", fake_call)

    chat._batch_generate_and_evaluate("a", "hi there", num_alternatives=2)
    assert len(calls) == 3


def test_quality_assessor_keys():
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))
    metrics = chat.quality_assessor.comprehensive_score("resp", "prompt")
    for key in ["relevance", "completeness", "clarity", "accuracy", "overall"]:
        assert key in metrics
