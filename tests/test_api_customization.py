import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from starlette.testclient import TestClient  # noqa: E402

import recthink_web_v2  # noqa: E402
from core.chat_v2 import ThinkingResult  # noqa: E402


def test_send_message_custom_rounds(monkeypatch):
    recthink_web_v2.chat_sessions.clear()
    captured = {}

    class DummyEngine:
        async def think_and_respond(
            self,
            message,
            thinking_rounds=None,
            alternatives_per_round=3,
        ):
            captured["rounds"] = thinking_rounds
            captured["alts"] = alternatives_per_round
            return ThinkingResult(
                response="ok",
                thinking_rounds=thinking_rounds or 0,
                thinking_history=[],
                total_tokens=1,
                processing_time=0.1,
                convergence_reason="done",
                metadata={},
            )

    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config: DummyEngine(),
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
    assert captured["rounds"] == 2
    assert captured["alts"] == 4
    assert resp.json()["thinking_rounds"] == 2


def test_send_message_defaults(monkeypatch):
    recthink_web_v2.chat_sessions.clear()
    captured = {}

    class DummyEngine:
        async def think_and_respond(
            self,
            message,
            thinking_rounds=None,
            alternatives_per_round=3,
        ):
            captured["rounds"] = thinking_rounds
            captured["alts"] = alternatives_per_round
            return ThinkingResult(
                response="ok",
                thinking_rounds=3,
                thinking_history=[],
                total_tokens=1,
                processing_time=0.1,
                convergence_reason="done",
                metadata={},
            )

    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config: DummyEngine(),
    )

    client = TestClient(recthink_web_v2.app)
    resp = client.post(
        "/chat",
        json={"session_id": "s2", "message": "hi"},
    )
    assert resp.status_code == 200
    assert captured["rounds"] is None
    assert captured["alts"] == 3
    assert resp.json()["thinking_rounds"] == 3
