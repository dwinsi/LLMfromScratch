from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Complete mutable state for one agent session."""

    messages: Annotated[list[BaseMessage], add_messages]
    context_paths: list[str]
    mode: str  # "default" | "explain" | "refactor" | "debug"
    session_id: str
