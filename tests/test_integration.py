import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import (  # noqa: E402
    EnhancedRecursiveThinkingChat,
    CoRTConfig,
)


@pytest.fixture
def cort_config():
    return CoRTConfig(api_key="x")


@pytest.fixture
def chat(cort_config):
    return EnhancedRecursiveThinkingChat(cort_config)


def test_convergence_detection(monkeypatch, chat):
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 3)
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "resp1")

    scores = {"resp1": 0.3, "resp2": 0.4, "resp3": 0.5}
    gen_calls = []

    def fake_batch(current_best, prompt, alts=3):
        if not gen_calls:
            gen_calls.append(1)
            return "resp2", ["resp2"], "improved"
        gen_calls.append(1)
        return "resp3", ["resp3"], "improved"

    def fake_score(resp, prompt):
        return {
            "relevance": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "accuracy": 0.0,
            "overall": scores[resp],
        }

    def fake_similarity(a, b):
        if {a, b} == {"resp2", "resp3"}:
            return 0.96
        return 0.5

    monkeypatch.setattr(chat, "_batch_generate_and_evaluate", fake_batch)
    monkeypatch.setattr(chat.quality_assessor, "comprehensive_score", fake_score)
    monkeypatch.setattr(chat, "_semantic_similarity", fake_similarity)

    result = chat.think_and_respond("hi", verbose=False)

    assert len(gen_calls) == 2
    rounds = [i for i in result.thinking_history if i.get("round") == 2]
    assert rounds and rounds[0]["response"] == "resp3"


def test_quality_plateau(monkeypatch, chat):
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 4)
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "resp1")

    scores = {"resp1": 0.5, "resp2": 0.52, "resp3": 0.528}
    gen_calls = []

    def fake_batch(current_best, prompt, alts=3):
        if not gen_calls:
            gen_calls.append(1)
            return "resp2", ["resp2"], "improved"
        gen_calls.append(1)
        return "resp3", ["resp3"], "improved"

    def fake_score(resp, prompt):
        return {
            "relevance": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "accuracy": 0.0,
            "overall": scores[resp],
        }

    monkeypatch.setattr(chat, "_batch_generate_and_evaluate", fake_batch)
    monkeypatch.setattr(chat.quality_assessor, "comprehensive_score", fake_score)
    monkeypatch.setattr(chat, "_semantic_similarity", lambda *a, **k: 0.5)

    result = chat.think_and_respond("hi", verbose=False)

    assert len(gen_calls) == 2
    rounds = [i for i in result.thinking_history if i.get("round") == 2]
    assert rounds and rounds[0]["response"] == "resp3"
