import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from core.chat import EnhancedRecursiveThinkingChat, CoRTConfig  # noqa: E402
from monitoring import MetricsRecorder  # noqa: E402


def test_metrics_recording(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 1)
    monkeypatch.setattr(chat, "_trim_conversation_history", lambda: None)

    def fake_call_api(messages, *a, **k):
        chat.full_thinking_log.append({"messages": messages, "response": "resp"})
        return "resp"

    monkeypatch.setattr(chat, "_call_api", fake_call_api)
    monkeypatch.setattr(
        chat,
        "_batch_generate_and_evaluate",
        lambda *a, **k: ("resp", ["resp"], "same"),
    )
    monkeypatch.setattr(chat.quality_assessor, "comprehensive_score", lambda *a, **k: {"overall": 1})
    monkeypatch.setattr(chat, "_semantic_similarity", lambda *a, **k: 0.0)

    recorder = MetricsRecorder()
    chat.think_and_respond("hi", verbose=False, metrics_recorder=recorder)
    assert len(recorder.runs) == 1
    run = recorder.runs[0]
    assert run.processing_time >= 0
    assert run.token_usage > 0
    assert run.num_rounds == 1
    assert run.convergence_reason

    summary = recorder.summary()
    assert summary["runs"] == 1
    assert "avg_processing_time" in summary


def test_summary_helper():
    rec = MetricsRecorder()
    rec.record_run(1.0, 10, 2, "converged")
    rec.record_run(2.0, 20, 3, "converged")
    text = rec.summary()
    assert text["runs"] == 2
