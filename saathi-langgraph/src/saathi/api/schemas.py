"""Pydantic request/response schemas for the Saathi API."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Thread/session ID — reuse to continue a conversation",
    )
    mode: Literal["default", "explain", "refactor", "debug"] = "default"
    context_paths: list[str] = Field(
        default=[],
        description="File or directory paths the agent should scope its context to",
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls_made: int = Field(
        default=0, description="Number of tool invocations during this turn"
    )


class StreamChunk(BaseModel):
    """One SSE data payload for a streaming chat response."""

    session_id: str
    delta: str
    done: bool = False


class SessionCreateRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Custom session ID; auto-generated if omitted",
    )
    mode: Literal["default", "explain", "refactor", "debug"] = "default"


class MessageRecord(BaseModel):
    role: Literal["human", "ai", "tool", "system"]
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    mode: str
    messages: list[MessageRecord]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    model: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model: str
    base_url: str
    temperature: float
    context_window: int
    max_tokens: int
    max_parallel_tools: int
