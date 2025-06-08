import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import recthink_web  # noqa: E402


def setup_session(client):
    resp = client.post("/api/initialize", json={"api_key": "x", "model": "m"})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def test_send_message_custom_rounds(monkeypatch):
    client = TestClient(recthink_web.app)
    session_id = setup_session(client)

    captured = {}

    async def fake_think(
        msg,
        verbose=True,
        thinking_rounds=None,
        alternatives_per_round=3,
    ):
        captured["rounds"] = thinking_rounds
        captured["alts"] = alternatives_per_round
        return {
            "response": "ok",
            "thinking_rounds": thinking_rounds,
            "thinking_history": [],
        }

    recthink_web.chat_instances[session_id].think_and_respond = fake_think

    resp = client.post(
        "/api/send_message",
        json={
            "session_id": session_id,
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
    client = TestClient(recthink_web.app)
    session_id = setup_session(client)

    captured = {}

    async def fake_think(
        msg,
        verbose=True,
        thinking_rounds=None,
        alternatives_per_round=3,
    ):
        captured["rounds"] = thinking_rounds
        captured["alts"] = alternatives_per_round
        return {
            "response": "ok",
            "thinking_rounds": 3,
            "thinking_history": [],
        }

    recthink_web.chat_instances[session_id].think_and_respond = fake_think

    resp = client.post(
        "/api/send_message",
        json={"session_id": session_id, "message": "hi"},
    )
    assert resp.status_code == 200
    assert captured["rounds"] is None
    assert captured["alts"] == 3
    assert resp.json()["thinking_rounds"] == 3
