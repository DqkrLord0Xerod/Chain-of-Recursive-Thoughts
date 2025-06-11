from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import settings
import uvicorn
import json
import os
from datetime import datetime
from typing import Optional
import logging

# Import the main RecThink class
from core.chat import AsyncEnhancedRecursiveThinkingChat, CoRTConfig

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
engine_instances: dict[str, AsyncEnhancedRecursiveThinkingChat] = {}


# Pydantic models for request/response validation
class ChatConfig(BaseModel):
    api_key: str | None = settings.openrouter_api_key
    model: str = settings.model


class MessageRequest(BaseModel):
    session_id: str
    message: str
    thinking_rounds: Optional[int] = None
    alternatives_per_round: Optional[int] = 3


class SaveRequest(BaseModel):
    session_id: str
    filename: Optional[str] = None


@app.post("/api/initialize")
async def initialize_chat(config: ChatConfig):
    """Initialize a new chat session"""
    try:
        # Generate a session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        
        # Initialize the engine instance
        engine = AsyncEnhancedRecursiveThinkingChat(
            CoRTConfig(api_key=config.api_key, model=config.model)
        )
        await engine.__aenter__()
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
        else:
            resp_text = result["response"]
            rounds = result["thinking_rounds"]
            history = result["thinking_history"]

        return {
            "session_id": request.session_id,
            "response": resp_text,
            "thinking_rounds": rounds,
            "thinking_history": history,
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

    engine = engine_instances.pop(session_id)
    await engine.close()
    return {"status": "deleted", "session_id": session_id}


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
                    "thinking_history": result.thinking_history,
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
