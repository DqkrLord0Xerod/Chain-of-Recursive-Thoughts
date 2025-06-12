import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
import recthink_web_v2  # noqa: E402
from core.chat_v2 import ThinkingResult, ThinkingRound  # noqa: E402


class DummyEngine:
    async def think_and_respond(self, message, thinking_rounds=None, alternatives_per_round=3):
        return ThinkingResult(
            response="ok",
            thinking_rounds=1,
            thinking_history=[
                ThinkingRound(
                    round_number=0,
                    response="ok",
                    alternatives=[],
                    selected=True,
                    explanation="init",
                    quality_score=0.5,
                    duration=0.0,
                )
            ],
            total_tokens=1,
            processing_time=0.1,
            convergence_reason="done",
            metadata={},
        )

    async def think_stream(self, message, context=None):
        yield {"stage": "start", "response": "partial", "quality": 0.1}
        yield {"stage": "end", "response": "final", "quality": 0.2}


def setup_module(module):
    recthink_web_v2.chat_sessions.clear()


def test_chat_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(recthink_web_v2, "create_optimized_engine", lambda config: DummyEngine())

    resp = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s1"
    assert data["response"] == "ok"
    assert data["thinking_rounds"] == 1


def test_websocket_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(recthink_web_v2, "create_optimized_engine", lambda config: DummyEngine())

    with client.websocket_connect("/ws/s2") as ws:
        ws.send_text(json.dumps({"message": "hello"}))
        data = ws.receive_json()
        assert data["type"] == "final"
        assert data["response"] == "ok"


def test_websocket_stream(monkeypatch):
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(recthink_web_v2, "create_optimized_engine", lambda config: DummyEngine())

    with client.websocket_connect("/ws/stream/s3") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        first = ws.receive_json()
        second = ws.receive_json()
        assert first["stage"] == "start"
        assert second["stage"] == "end"


def test_websocket_stream_order(monkeypatch):
    """Ensure streaming messages arrive in the expected order."""
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config: DummyEngine(),
    )

    with client.websocket_connect("/ws/stream/s4") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        updates = [ws.receive_json() for _ in range(2)]
        stages = [u["stage"] for u in updates]
        assert stages == ["start", "end"]
