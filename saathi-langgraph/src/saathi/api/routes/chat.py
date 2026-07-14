"""Chat endpoints: POST /chat (full response) and POST /chat/stream (SSE)."""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from saathi.api.dependencies import GraphDep
from saathi.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


def _build_input(req: ChatRequest) -> dict:
    return {
        "messages": [HumanMessage(content=req.message)],
        "context_paths": req.context_paths,
        "mode": req.mode,
        "session_id": req.session_id,
    }


def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _count_tool_calls(messages: list) -> int:
    return sum(
        len(m.tool_calls)
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: GraphDep) -> ChatResponse:
    """
    Send a message and receive the agent's full reply.

    The agent runs the complete ReAct loop (including tool calls) before
    responding, so this may take a few seconds for complex tasks.
    """
    result = await graph.ainvoke(
        _build_input(req),
        config=_thread_config(req.session_id),
    )

    messages = result.get("messages", [])
    # Last AI message is the final answer
    reply = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        tool_calls_made=_count_tool_calls(messages),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, graph: GraphDep):
    """
    Stream the agent's reply token-by-token using Server-Sent Events.

    Each SSE event carries a JSON payload:
      {"session_id": "...", "delta": "<token>", "done": false}

    The final event has `"done": true` and an empty `delta`.

    Connect with an EventSource or `curl -N`:
      curl -N -X POST http://localhost:8000/chat/stream \\
           -H "Content-Type: application/json" \\
           -d '{"message": "explain this repo"}'
    """

    async def event_generator():
        async for chunk, _metadata in graph.astream(
            _build_input(req),
            config=_thread_config(req.session_id),
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                payload = json.dumps(
                    {"session_id": req.session_id, "delta": delta, "done": False}
                )
                yield f"data: {payload}\n\n"

        # Terminal event
        yield f"data: {json.dumps({'session_id': req.session_id, 'delta': '', 'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )
