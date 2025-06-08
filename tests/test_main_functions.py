import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import recursive_thinking_ai  # noqa: E402
from recursive_thinking_ai import (  # noqa: E402
    EnhancedRecursiveThinkingChat,
    ConvergenceTracker,
    CoRTConfig,
)


def test_determine_rounds_valid(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "4")
    assert chat._determine_thinking_rounds("prompt") == 4


def test_determine_rounds_bounds(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "7")
    assert chat._determine_thinking_rounds("prompt") == 5


def test_think_and_respond_flow(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 1)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "initial")
    monkeypatch.setattr(
        chat,
        "_batch_generate_and_evaluate",
        lambda *a, **k: ("alt1", ["alt1"], "pick alt"),
    )
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)

    result = chat.think_and_respond("hi", verbose=False)

    assert result.response == "alt1"
    assert result.thinking_rounds == 1
    assert chat.conversation_history[-1]["content"] == "alt1"
    assert any(
        item.get("selected")
        for item in result.thinking_history
        if item["response"] == "alt1"
    )
    assert hasattr(result, "api_calls")
    assert hasattr(result, "processing_time")


def test_convergence_tracker():
    tracker = ConvergenceTracker(
        lambda a, b: 0.96,
        lambda resp, prompt: 0.5 if resp == "prev" else 0.51,
    )
    tracker.add("prev", "p")
    tracker.add("new", "p")
    cont, reason = tracker.should_continue("p")
    assert not cont
    assert reason == "converged"

    tracker = ConvergenceTracker(
        lambda a, b: 0.2,
        lambda resp, prompt: 0.2 if resp == "prev" else 0.5,
    )
    tracker.add("prev", "p")
    tracker.add("new", "p")
    cont, reason = tracker.should_continue("p")
    assert cont
    assert reason == "continue"


def test_convergence_tracker_oscillation():
    vals = iter([0.0, 0.05, 0.1])

    def score_fn(resp, prompt):
        return next(vals)

    def sim_fn(a, b):
        if a == b:
            return 1.0
        return 0.2

    tracker = ConvergenceTracker(sim_fn, score_fn)
    tracker.add("a", "p")
    tracker.add("b", "p")
    tracker.add("a", "p")
    cont, reason = tracker.should_continue("p")
    assert not cont
    assert reason == "oscillation"


def test_convergence_tracker_update():
    tracker = ConvergenceTracker(
        lambda a, b: 0.96,
        lambda resp, prompt: 0.5 if resp == "prev" else 0.51,
    )
    tracker.add("prev", "p")
    cont, reason = tracker.update("new", "p")
    assert not cont
    assert reason == "converged"


def test_think_and_respond_early_stop(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="test"))
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 3)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "initial")

    gen_calls = []

    def fake_batch(*a, **k):
        gen_calls.append(1)
        return "initial", ["alt1"], "same"

    monkeypatch.setattr(chat, "_batch_generate_and_evaluate", fake_batch)

    class DummyTracker:
        def add(self, r, p):
            pass

        def should_continue(self, p):
            return False, "stop"

    monkeypatch.setattr(
        recursive_thinking_ai, "ConvergenceTracker", lambda *a, **k: DummyTracker()
    )
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)

    result = chat.think_and_respond("hi", verbose=False)

    assert len(gen_calls) == 1
    assert len([i for i in result.thinking_history if i.get("round") == 1]) == 1
    assert hasattr(result, "api_calls")
    assert hasattr(result, "processing_time")
