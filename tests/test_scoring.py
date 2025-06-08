import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def test_score_called(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    calls = []

    def fake_score(resp, prompt):
        calls.append(resp)
        return 0.5

    def fake_call(messages, temperature=0.7, stream=True):
        return json.dumps({"alternatives": ["b", "c"], "choice": "1"})

    monkeypatch.setattr(chat, "_score_response", fake_score)
    monkeypatch.setattr(chat, "_call_api", fake_call)

    chat._batch_generate_and_evaluate("a", "hi there", num_alternatives=2)
    assert len(calls) == 3
