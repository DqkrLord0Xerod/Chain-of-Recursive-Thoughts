from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import settings
from api import fetch_models
import uvicorn
import json
import os
from datetime import datetime
from typing import Optional
from dataclasses import asdict
import logging

# Import the main RecThink class
from core.chat_v2 import (
    CoRTConfig,
    create_default_engine,
    RecursiveThinkingEngine,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RecThink API", description="API for Enhanced Recursive Thinking Chat")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dictionary to store engine instances
engine_instances: dict[str, RecursiveThinkingEngine] = {}


# Pydantic models for request/response validation
class ChatConfig(BaseModel):
    api_key: str | None = settings.openrouter_api_key
    model: str = settings.model
    budget_token_limit: int = 100000
    enforce_budget: bool = True


class MessageRequest(BaseModel):
    session_id: str
    message: str
    thinking_rounds: Optional[int] = None
    alternatives_per_round: Optional[int] = 3


class SaveRequest(BaseModel):
    session_id: str
    filename: Optional[str] = None


class FinalizeRequest(BaseModel):
    session_id: str


@app.post("/api/initialize")
async def initialize_chat(config: ChatConfig):
    """Initialize a new chat session"""
    try:
        # Generate a session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        
        # Initialize the engine instance
        budget_limit = (
            config.budget_token_limit
            if config.enforce_budget
            else 1_000_000_000
        )
        engine = create_default_engine(
            CoRTConfig(
                api_key=config.api_key,
                model=config.model,
                budget_token_limit=budget_limit,
            )
        )
        engine_instances[session_id] = engine
        
        return {"session_id": session_id, "status": "initialized"}
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize chat: {str(e)}")


@app.post("/api/send_message")
async def send_message(request: MessageRequest):
    """Send a message and get a response with thinking process"""
    try:
        if request.session_id not in engine_instances:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = engine_instances[request.session_id]

        result = await engine.think_and_respond(
            request.message,
            thinking_rounds=request.thinking_rounds,
            alternatives_per_round=request.alternatives_per_round,
        )
        
        if hasattr(result, "response"):
            resp_text = result.response
            rounds = result.thinking_rounds
            history = result.thinking_history
            cost_total = result.cost_total
            cost_this_step = result.cost_this_step
        else:
            resp_text = result["response"]
            rounds = result["thinking_rounds"]
            history = result["thinking_history"]
            cost_total = result.get("cost_total", 0)
            cost_this_step = result.get("cost_this_step", 0)

        return {
            "session_id": request.session_id,
            "response": resp_text,
            "thinking_rounds": rounds,
            "thinking_history": history,
            "cost_total": cost_total,
            "cost_this_step": cost_this_step,
        }
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@app.post("/api/save")
async def save_conversation(request: SaveRequest):
    """Save the conversation or full thinking log"""
    try:
        if request.session_id not in engine_instances:
            raise HTTPException(status_code=404, detail="Session not found")

        engine = engine_instances[request.session_id]

        filename = request.filename or "conversation.json"
        await engine.save_conversation(filename)

        return {"status": "saved", "filename": filename}
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation: {str(e)}")


@app.get("/api/models")
async def list_models():
    """Return available model metadata."""
    try:
        return {"models": fetch_models()}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")


@app.get("/api/sessions")
async def list_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, engine in engine_instances.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(engine.conversation_history) // 2,
            "created_at": session_id.split("_")[1]  # Extract timestamp from session ID
        })
    
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in engine_instances:
        raise HTTPException(status_code=404, detail="Session not found")

    engine_instances.pop(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/cost/{session_id}")
async def get_cost(session_id: str):
    """Return current budget usage for a session."""
    if session_id not in engine_instances:
        raise HTTPException(status_code=404, detail="Session not found")

    manager = engine_instances[session_id].budget_manager
    if manager is None:
        return {
            "session_id": session_id,
            "token_limit": 0,
            "tokens_used": 0,
            "dollars_spent": 0.0,
        }

    return {
        "session_id": session_id,
        "token_limit": manager.token_limit,
        "tokens_used": manager.tokens_used,
        "dollars_spent": manager.dollars_spent,
    }


@app.post("/api/finalize")
async def finalize_session(request: FinalizeRequest):
    """Summarize conversation and end the session."""
    if request.session_id not in engine_instances:
        raise HTTPException(status_code=404, detail="Session not found")

    engine = engine_instances.pop(request.session_id)

    summary = await engine.summarize_history()

    manager = engine.budget_manager
    cost = {
        "token_limit": manager.token_limit if manager else 0,
        "tokens_used": manager.tokens_used if manager else 0,
        "dollars_spent": manager.dollars_spent if manager else 0.0,
    }

    return {"summary": summary, "cost": cost}


# WebSocket for streaming thinking process
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in engine_instances:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    engine = engine_instances[session_id]

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data["type"] == "message":
                result = await engine.think_and_respond(message_data["content"])
                await websocket.send_json({
                    "type": "final",
                    "response": result.response,
                    "thinking_rounds": result.thinking_rounds,
                    "thinking_history": [
                        asdict(r) for r in result.thinking_history
                    ],
                    "cost_total": result.cost_total,
                    "cost_this_step": result.cost_this_step,
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})


# Serve the React app
@app.get("/")
async def root():
    return {"message": f"RecThink API is running. Frontend available at {settings.frontend_url}"}

if __name__ == "__main__":
    uvicorn.run("recthink_web:app", host="0.0.0.0", port=8000, reload=True)
