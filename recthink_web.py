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

# Create a dictionary to store chat instances
chat_instances = {}


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
    full_log: bool = False


@app.post("/api/initialize")
async def initialize_chat(config: ChatConfig):
    """Initialize a new chat session"""
    try:
        # Generate a session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        
        # Initialize the chat instance
        chat = AsyncEnhancedRecursiveThinkingChat(
            CoRTConfig(api_key=config.api_key, model=config.model)
        )
        chat_instances[session_id] = chat
        
        return {"session_id": session_id, "status": "initialized"}
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize chat: {str(e)}")


@app.post("/api/send_message")
async def send_message(request: MessageRequest):
    """Send a message and get a response with thinking process"""
    try:
        if request.session_id not in chat_instances:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat = chat_instances[request.session_id]

        result = await chat.think_and_respond(
            request.message,
            verbose=True,
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
        if request.session_id not in chat_instances:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat = chat_instances[request.session_id]
        
        filename = request.filename
        if request.full_log:
            chat.save_full_log(filename)
        else:
            chat.save_conversation(filename)
        
        return {"status": "saved", "filename": filename}
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation: {str(e)}")


@app.get("/api/sessions")
async def list_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, chat in chat_instances.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(chat.conversation_history) // 2,  # Each message-response pair counts as 2
            "created_at": session_id.split("_")[1]  # Extract timestamp from session ID
        })
    
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_instances[session_id]
    return {"status": "deleted", "session_id": session_id}


# WebSocket for streaming thinking process
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in chat_instances:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    chat = chat_instances[session_id]
    
    try:
        # Set up a custom callback to stream thinking process
        original_call_api = chat._async_call_api
        
        async def stream_callback(chunk):
            await websocket.send_json({"type": "chunk", "content": chunk})
        
        # Override the _call_api method to also send updates via WebSocket
        async def ws_call_api(messages, temperature=0.7, stream=True):
            result = await original_call_api(messages, temperature)
            if stream:
                await stream_callback(result)
            return result
        
        # Replace the method temporarily
        chat._async_call_api = ws_call_api
        
        # Wait for messages from the client
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                # Process the message
                result = await chat.think_and_respond(message_data["content"], verbose=True)
                
                # Send the final result
                if hasattr(result, "response"):
                    resp_text = result.response
                    rounds = result.thinking_rounds
                    history = result.thinking_history
                else:
                    resp_text = result["response"]
                    rounds = result["thinking_rounds"]
                    history = result["thinking_history"]

                await websocket.send_json({
                    "type": "final",
                    "response": resp_text,
                    "thinking_rounds": rounds,
                    "thinking_history": history,
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})
    finally:
        # Restore original method
        chat._async_call_api = original_call_api


# Serve the React app
@app.get("/")
async def root():
    return {"message": f"RecThink API is running. Frontend available at {settings.frontend_url}"}

if __name__ == "__main__":
    uvicorn.run("recthink_web:app", host="0.0.0.0", port=8000, reload=True)
