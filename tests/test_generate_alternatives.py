import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def test_generate_alternatives_single_call(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    calls = []

    def fake_call(messages, temperature=0.7, stream=True):
        calls.append(messages)
        return json.dumps({"alternatives": ["a1", "a2"]})

    monkeypatch.setattr(chat, "_call_api", fake_call)
    alts = chat._generate_alternatives("base", "prompt", num_alternatives=2)

    assert len(calls) == 1
    assert alts == ["a1", "a2"]
