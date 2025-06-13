import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
import recthink_web_v2  # noqa: E402
from core.chat_v2 import ThinkingResult, ThinkingRound  # noqa: E402
from core.loop_controller import LoopState  # noqa: E402


class DummyEngine:
    async def think_and_respond(
        self,
        message,
        thinking_rounds=None,
        alternatives_per_round=3,
        session_id=None,
    ):
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
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config, router=None, budget_manager=None: DummyEngine(),
    )

    resp = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s1"
    assert data["response"] == "ok"
    assert data["thinking_rounds"] == 1


def test_websocket_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config, router=None, budget_manager=None: DummyEngine(),
    )

    with client.websocket_connect("/ws/s2") as ws:
        ws.send_text(json.dumps({"message": "hello"}))
        data = ws.receive_json()
        assert data["type"] == "final"
        assert data["response"] == "ok"


def test_websocket_stream(monkeypatch):
    client = TestClient(recthink_web_v2.app)
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config, router=None, budget_manager=None: DummyEngine(),
    )

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
        lambda config, router=None, budget_manager=None: DummyEngine(),
    )

    with client.websocket_connect("/ws/stream/s4") as ws:
        ws.send_text(json.dumps({"message": "hi"}))
        updates = [ws.receive_json() for _ in range(2)]
        stages = [u["stage"] for u in updates]
        assert stages == ["start", "end"]


def test_provider_health_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)

    class DummyAnalyzer:
        def get_provider_health(self):
            return [{"provider": "p1", "status": "healthy"}]

    monkeypatch.setattr(recthink_web_v2, "metrics_analyzer", DummyAnalyzer())

    resp = client.get("/health/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["providers"][0]["provider"] == "p1"


def test_cache_health_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)

    class DummyCache:
        async def stats(self):
            return {"type": "memory", "entries": 1}

    class DummyEngine:
        def __init__(self):
            self.cache = DummyCache()

    recthink_web_v2.chat_sessions["s1"] = DummyEngine()

    resp = client.get("/health/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert data["caches"][0]["session_id"] == "s1"
    assert data["caches"][0]["type"] == "memory"

    recthink_web_v2.chat_sessions.clear()


def test_batch_chat_endpoint(monkeypatch):
    client = TestClient(recthink_web_v2.app)

    class DummyBatchOpt:
        async def think_batch(self, prompts):
            return [(p + "-ok", {}) for p in prompts]

    class EngineWithParallel:
        def __init__(self):
            self.parallel_optimizer = object()

    monkeypatch.setattr(
        recthink_web_v2,
        "BatchThinkingOptimizer",
        lambda opt: DummyBatchOpt(),
    )
    monkeypatch.setattr(
        recthink_web_v2,
        "create_optimized_engine",
        lambda config, router=None, budget_manager=None: EngineWithParallel(),
    )

    class DummyAnalyzer:
        def record_batch(self, size, duration):
            self.last = (size, duration)

    analyzer = DummyAnalyzer()
    monkeypatch.setattr(recthink_web_v2, "metrics_analyzer", analyzer)

    resp = client.post(
        "/chat/batch",
        json={"session_id": "s5", "messages": ["x", "y"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["responses"] == ["x-ok", "y-ok"]
    assert analyzer.last[0] == 2


def test_history_endpoints(monkeypatch):
    client = TestClient(recthink_web_v2.app)

    class DummyController:
        async def load_loop_history(self, session_id):
            return [
                LoopState(
                    rounds=[],
                    scores=[0.5],
                    convergence_reason="done",
                    start_time=0.0,
                    end_time=1.0,
                )
            ]

        async def get_convergence_reasons(self, session_id):
            return ["done"]

    monkeypatch.setattr(recthink_web_v2, "LoopController", lambda *_: DummyController())

    resp = client.get("/sessions/s1/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s1"
    assert len(data["history"]) == 1

    resp = client.get("/sessions/s1/convergence")
    assert resp.status_code == 200
    assert resp.json()["reasons"] == ["done"]
