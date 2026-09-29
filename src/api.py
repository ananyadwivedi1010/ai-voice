"""
api.py — FastAPI server for the voice agent.

Endpoints:
    POST /chat    — Get agent reply given version and history
    GET  /health  — Health check

Usage:
    uvicorn src.api:app --reload --port 8000
"""

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.agent import get_agent_reply

load_dotenv()

app = FastAPI(
    title="Hinglish Voice Agent API",
    description="API for Annie, the QuickCash personal loan sales agent",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Authentication (optional but recommended for production)
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> None:
    """
    Verify the API key from the X-API-Key header.
    Set API_SECRET_KEY in .env to enable auth.
    """
    expected_key = os.environ.get("API_SECRET_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    version: Literal["v1", "v2", "v3"] = Field(
        ..., description="Prompt version to use"
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Conversation history in Claude format",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "version": "v3",
                "history": [
                    {"role": "user", "content": "Haan, kaun hai?"},
                    {"role": "assistant", "content": "Namaste! Main Annie bol rahi hoon..."},
                ],
            }
        }


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's next reply")


class HealthResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check if the API is running."""
    return HealthResponse(status="ok", message="Voice agent API is running")


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(
    request: ChatRequest,
    _: None = Security(verify_api_key),
) -> ChatResponse:
    """
    Get the agent's next reply given a conversation history.

    The history should be in Claude's message format:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Returns the agent's next reply as plain text.
    """
    try:
        # Convert pydantic models to dicts
        history = [msg.model_dump() for msg in request.history]

        reply = get_agent_reply(version=request.version, history=history)
        return ChatResponse(reply=reply)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating reply: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# CORS (uncomment if needed for browser clients)
# ---------------------------------------------------------------------------
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
