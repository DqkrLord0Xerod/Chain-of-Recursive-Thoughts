import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def test_determine_rounds_valid(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "4")
    assert chat._determine_thinking_rounds("prompt") == 4


def test_determine_rounds_bounds(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "7")
    assert chat._determine_thinking_rounds("prompt") == 5


def test_generate_alternatives_fallback(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "a1\na2\n")
    alts = chat._generate_alternatives("base", "prompt", num_alternatives=2)
    assert alts == ["a1", "a2"]


def test_think_and_respond_flow(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 1)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "initial")
    monkeypatch.setattr(
        chat,
        "_generate_alternatives",
        lambda *a, **k: ["alt1"],
    )
    monkeypatch.setattr(
        chat,
        "_evaluate_responses",
        lambda *a, **k: ("alt1", "pick alt"),
    )
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)

    result = chat.think_and_respond("hi", verbose=False)

    assert result["response"] == "alt1"
    assert result["thinking_rounds"] == 1
    assert chat.conversation_history[-1]["content"] == "alt1"
    assert any(
        item.get("selected")
        for item in result["thinking_history"]
        if item["response"] == "alt1"
    )
