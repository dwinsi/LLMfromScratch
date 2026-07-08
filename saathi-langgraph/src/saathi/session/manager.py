"""Session save/load backed by JSON files in .saathi/sessions/."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass
class SessionState:
    model_id: str
    context_paths: list[str] = field(default_factory=list)
    mode: str = "default"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    last_response: str = ""


class SessionManager:
    _root = Path(".saathi") / "sessions"

    def save(self, name: str, state: SessionState, messages: list[BaseMessage]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(),
            "state": {
                "model_id": state.model_id,
                "context_paths": state.context_paths,
                "mode": state.mode,
                "session_id": state.session_id,
            },
            "messages": [_serialize_message(m) for m in messages],
        }
        path = self._root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, name: str) -> tuple[SessionState, list[BaseMessage]]:
        path = self._root / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{name}' not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["state"]
        state = SessionState(
            model_id=s["model_id"],
            context_paths=s.get("context_paths", []),
            mode=s.get("mode", "default"),
            session_id=s.get("session_id", str(uuid.uuid4())),
        )
        messages = [_deserialize_message(m) for m in data.get("messages", [])]
        return state, messages

    def list_sessions(self) -> list[tuple[str, str]]:
        if not self._root.exists():
            return []
        sessions = []
        for p in sorted(self._root.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                saved_at = data.get("saved_at", "unknown")
            except (json.JSONDecodeError, OSError):
                saved_at = "unknown"
            sessions.append((p.stem, saved_at))
        return sessions


def _serialize_message(msg: BaseMessage) -> dict:
    return {"type": msg.__class__.__name__, "content": msg.content}


def _deserialize_message(data: dict) -> BaseMessage:
    t = data.get("type", "HumanMessage")
    content = data.get("content", "")
    mapping: dict[str, type[BaseMessage]] = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
    }
    cls = mapping.get(t, HumanMessage)
    return cls(content=content)
