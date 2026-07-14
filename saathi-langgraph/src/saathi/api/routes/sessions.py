"""Session management endpoints."""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from saathi.api.dependencies import GraphDep
from saathi.api.schemas import MessageRecord, SessionCreateRequest, SessionHistoryResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "human"
    if isinstance(msg, AIMessage):
        return "ai"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, SystemMessage):
        return "system"
    return "ai"


@router.post("", response_model=SessionHistoryResponse, status_code=201)
async def create_session(body: SessionCreateRequest, graph: GraphDep) -> SessionHistoryResponse:
    """
    Create (or reset) a named session.

    Sending the first real message via POST /chat with this session_id will
    initialise checkpoint state automatically. This endpoint simply confirms
    the session ID is valid and returns an empty history.
    """
    return SessionHistoryResponse(
        session_id=body.session_id,
        mode=body.mode,
        messages=[],
    )


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, graph: GraphDep) -> SessionHistoryResponse:
    """Retrieve conversation history for an existing session."""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)

    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    state = snapshot.values
    raw_messages = state.get("messages", [])
    mode = state.get("mode", "default")

    records = [
        MessageRecord(
            role=_role(m),
            content=m.content if isinstance(m.content, str) else str(m.content),
        )
        for m in raw_messages
    ]

    return SessionHistoryResponse(session_id=session_id, mode=mode, messages=records)
