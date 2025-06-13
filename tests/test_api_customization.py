import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402

import recthink_web_v2  # noqa: E402
from core.chat_v2 import ThinkingResult  # noqa: E402


class DummyEngine:
    def __init__(self):
        self.captured = {}

    async def think_and_respond(
        self,
        message,
        thinking_rounds=None,
        alternatives_per_round=3,
        session_id=None,
    ):
        self.captured["rounds"] = thinking_rounds
        self.captured["alts"] = alternatives_per_round
        return ThinkingResult(
            response="ok",
            thinking_rounds=thinking_rounds if thinking_rounds is not None else 3,
            thinking_history=[],
            total_tokens=0,
            processing_time=0.0,
            convergence_reason="done",
            metadata={},
        )


def test_send_message_custom_rounds(monkeypatch):
    engine = DummyEngine()
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config: engine,
    )

    client = TestClient(recthink_web_v2.app)

    resp = client.post(
        "/chat",
        json={
            "session_id": "s1",
            "message": "hi",
            "thinking_rounds": 2,
            "alternatives_per_round": 4,
        },
    )
    assert resp.status_code == 200
    assert engine.captured["rounds"] == 2
    assert engine.captured["alts"] == 4
    assert resp.json()["thinking_rounds"] == 2


def test_send_message_defaults(monkeypatch):
    engine = DummyEngine()
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config: engine,
    )

    client = TestClient(recthink_web_v2.app)

    resp = client.post(
        "/chat",
        json={"session_id": "s2", "message": "hi"},
    )
    assert resp.status_code == 200
    assert engine.captured["rounds"] is None
    assert engine.captured["alts"] == 3
    assert resp.json()["thinking_rounds"] == 3
