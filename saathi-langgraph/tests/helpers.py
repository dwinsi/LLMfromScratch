"""Reusable helpers for building graph state in tests."""

from langchain_core.messages import AIMessage


def ai_with_tool_calls(calls: list[dict]) -> dict:
    """Build a graph state whose last message carries the given tool calls."""
    return {
        "messages": [AIMessage(content="", tool_calls=calls)],
        "context_paths": [],
        "mode": "default",
        "session_id": "test",
    }


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}
