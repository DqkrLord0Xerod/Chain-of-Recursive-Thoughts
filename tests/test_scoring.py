import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def test_score_called(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    calls = []

    def fake_score(resp, prompt):
        calls.append(resp)
        return 0.5

    def fake_call(messages, temperature=0.2, stream=True):
        return "1\nreason"

    monkeypatch.setattr(chat, "_score_response", fake_score)
    monkeypatch.setattr(chat, "_call_api", fake_call)

    chat._evaluate_responses("hi there", "a", ["b", "c"])
    assert len(calls) == 3
