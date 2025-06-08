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


def test_should_continue_thinking(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")

    monkeypatch.setattr(chat, "_semantic_similarity", lambda a, b: 0.96)
    monkeypatch.setattr(chat, "_score_response", lambda resp, prompt: 0.5 if resp == "prev" else 0.51)

    assert not chat._should_continue_thinking("prev", "new", "p")

    monkeypatch.setattr(chat, "_semantic_similarity", lambda a, b: 0.2)
    monkeypatch.setattr(chat, "_score_response", lambda resp, prompt: 0.2 if resp == "prev" else 0.5)

    assert chat._should_continue_thinking("prev", "new", "p")


def test_think_and_respond_early_stop(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 3)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "initial")

    gen_calls = []

    def fake_gen(*a, **k):
        gen_calls.append(1)
        return ["alt1"]

    monkeypatch.setattr(chat, "_generate_alternatives", fake_gen)
    monkeypatch.setattr(chat, "_evaluate_responses", lambda *a, **k: ("initial", "same"))
    monkeypatch.setattr(chat, "_should_continue_thinking", lambda *a, **k: False)
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)

    result = chat.think_and_respond("hi", verbose=False)

    assert len(gen_calls) == 1
    assert len([i for i in result["thinking_history"] if i.get("round") == 1]) == 1
