from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, Optional, List

import structlog

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.chat_v2 import CoRTConfig, ThinkingResult
from core.recursive_engine_v2 import (
    OptimizedRecursiveEngine,
    create_optimized_engine,
)
from core.optimization.parallel_thinking import BatchThinkingOptimizer
from monitoring.metrics_v2 import MetricsAnalyzer, ThinkingMetrics
from monitoring.telemetry import initialize_telemetry, instrument_fastapi
from config.config import load_production_config

app = FastAPI(title="RecThink API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = structlog.get_logger(__name__)

# Global store for chat engines per session
chat_sessions: Dict[str, "OptimizedRecursiveEngine"] = {}
metrics_analyzer = MetricsAnalyzer()


@app.on_event("startup")
async def init_telemetry() -> None:
    """Initialize telemetry using production configuration."""
    try:
        cfg = load_production_config()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("config_load_failed", error=str(exc))
        initialize_telemetry(enable_prometheus=False)
        return

    initialize_telemetry(
        service_name=cfg.app_name,
        service_version=cfg.app_version,
        enable_prometheus=cfg.monitoring.metrics_enabled,
        prometheus_port=cfg.monitoring.prometheus_port,
        jaeger_endpoint=cfg.monitoring.jaeger_endpoint,
    )
    instrument_fastapi(app)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    thinking_rounds: Optional[int] = None
    alternatives_per_round: int = 3


class BatchChatRequest(BaseModel):
    session_id: str
    messages: List[str]
    thinking_rounds: Optional[int] = None
    alternatives_per_round: int = 3


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = create_optimized_engine(CoRTConfig())

    engine = chat_sessions[request.session_id]
    start_time = time.time()
    try:
        result: ThinkingResult = await engine.think_and_respond(
            request.message,
            thinking_rounds=request.thinking_rounds,
            alternatives_per_round=request.alternatives_per_round,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    end_time = time.time()
    metrics = ThinkingMetrics(
        session_id=request.session_id,
        start_time=start_time,
        end_time=end_time,
        rounds_completed=result.thinking_rounds,
        convergence_reason=result.convergence_reason,
        quality_scores=[r.quality_score for r in result.thinking_history],
        round_durations=[r.duration for r in result.thinking_history],
        alternatives_generated=[len(r.alternatives) for r in result.thinking_history],
    )
    metrics_analyzer.record_session(metrics)

    history = [asdict(round) for round in result.thinking_history]

    return {
        "session_id": request.session_id,
        "response": result.response,
        "thinking_rounds": result.thinking_rounds,
        "thinking_history": history,
    }


@app.post("/chat/batch")
async def chat_batch_endpoint(request: BatchChatRequest):
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = create_optimized_engine(CoRTConfig())

    engine = chat_sessions[request.session_id]
    if not engine.parallel_optimizer:
        raise HTTPException(status_code=400, detail="Parallel thinking disabled")

    batch_opt = BatchThinkingOptimizer(engine.parallel_optimizer)

    start = time.time()
    try:
        results = await batch_opt.think_batch(request.messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    end = time.time()

    metrics_analyzer.record_batch(len(request.messages), end - start)

    responses = [r[0] for r in results]
    return {
        "session_id": request.session_id,
        "responses": responses,
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in chat_sessions:
        chat_sessions[session_id] = create_optimized_engine(CoRTConfig())

    engine = chat_sessions[session_id]

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                message = payload.get("message")
                rounds = payload.get("thinking_rounds")
                alts = payload.get("alternatives_per_round", 3)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            try:
                result: ThinkingResult = await engine.think_and_respond(
                    message,
                    thinking_rounds=rounds,
                    alternatives_per_round=alts,
                )
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            await websocket.send_json(
                {
                    "type": "final",
                    "response": result.response,
                    "thinking_rounds": result.thinking_rounds,
                    "thinking_history": [asdict(r) for r in result.thinking_history],
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


@app.get("/health/providers")
async def provider_health() -> Dict[str, List[Dict[str, object]]]:
    """Return health information for LLM providers."""
    health = metrics_analyzer.get_provider_health()
    logger.info("provider_health_status", providers=health)
    return {"providers": health}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Basic health check for service availability."""
    return {"status": "ok"}


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """Stream thinking updates using think_stream."""
    await websocket.accept()

    if session_id not in chat_sessions:
        chat_sessions[session_id] = create_optimized_engine(CoRTConfig())

    engine = chat_sessions[session_id]

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                message = payload.get("message")
                context = payload.get("context")
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            try:
                async for update in engine.think_stream(
                    message,
                    context=context,
                ):
                    await websocket.send_json(
                        {
                            "stage": update.get("stage"),
                            "response": update.get("response"),
                            "quality": update.get("quality"),
                        }
                    )
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


@app.get("/metrics/summary")
async def metrics_summary() -> Dict[str, object]:
    """Return summary statistics and recent anomalies."""
    latency = {}
    if hasattr(metrics_analyzer, "stage_latency"):
        latency = {stage: list(v) for stage, v in metrics_analyzer.stage_latency.items()}

    return {
        "summary": metrics_analyzer.get_summary_stats(),
        "anomalies": list(metrics_analyzer.anomalies),
        "stage_latency": latency,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("recthink_web_v2:app", host="0.0.0.0", port=8000, reload=True)
