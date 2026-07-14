# Chapter 12 — Testing FastAPI Apps

> **What you'll learn:** `TestClient` for sync tests, `AsyncClient` for async tests, overriding dependencies for mocking, `pytest-asyncio`, and practical test examples for the Saathi health, chat, and session endpoints.

---

## Why testing a FastAPI app is easy

FastAPI is built on Starlette, which ships a `TestClient` that runs your ASGI app in-process — no network, no port, no `uvicorn` required. You call it like a requests session. Your app starts up, handles the request, and returns a response — all in the same process, synchronously.

For async tests you use `httpx.AsyncClient` with an ASGI transport. Both approaches let you test the real routing, validation, and serialisation logic without a running server.

---

## Setup

```bash
pip install pytest pytest-asyncio httpx
```

Add to `pyproject.toml` (already done in Saathi):

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
```

`asyncio_mode = "auto"` means `async def test_*` functions are automatically treated as async tests — no `@pytest.mark.asyncio` decorator needed on each one.

---

## `TestClient` — synchronous testing

```python
from fastapi.testclient import TestClient
from saathi.api.main import app

client = TestClient(app)

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "running" in resp.json()["message"]
```

`TestClient` wraps `app` in a way that runs the lifespan events. If your lifespan tries to connect to Ollama or SQLite, it will actually do that — you'll need either a real Ollama running or dependency overrides (see below).

> **Experienced note:** This is equivalent to Flask's `app.test_client()` and Django's `APIClient`. The interface is requests-compatible: `.get()`, `.post()`, `.json()`, etc.

---

## Overriding dependencies for mocking

The most powerful testing technique in FastAPI is `dependency_overrides`. You can swap any `Depends()` dependency with a mock without touching the route code:

```python
from saathi.api.dependencies import get_graph
from saathi.api.main import app
from fastapi.testclient import TestClient

class MockGraph:
    """Fake graph that returns a canned response without calling Ollama."""
    async def ainvoke(self, input, config=None):
        from langchain_core.messages import AIMessage, HumanMessage
        return {
            "messages": [
                HumanMessage(content=input["messages"][0].content),
                AIMessage(content="This is a mock response."),
            ],
            "mode": "default",
            "session_id": input["session_id"],
            "context_paths": [],
        }

    async def aget_state(self, config):
        return None  # session not found

app.dependency_overrides[get_graph] = lambda: MockGraph()
client = TestClient(app)
```

Now `get_graph()` returns `MockGraph()` instead of the real compiled LangGraph. Every test that uses `client` runs against the mock — no Ollama, no SQLite, instant.

---

## Testing the health endpoint

```python
# tests/test_health.py
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from saathi.api.main import app

# Override get_graph so lifespan doesn't need Ollama
from saathi.api.dependencies import get_graph
app.dependency_overrides[get_graph] = lambda: object()

client = TestClient(app)

def test_health_ok():
    """Health check returns ok when Ollama is reachable."""
    with patch("saathi.api.routes.health.httpx.AsyncClient") as mock_client:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)

        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ollama_reachable"] is True

def test_health_degraded_when_ollama_down():
    """Health check returns degraded when Ollama connection refused."""
    import httpx
    with patch("saathi.api.routes.health.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        resp = client.get("/health")

    assert resp.status_code == 200   # health endpoint itself succeeds
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["ollama_reachable"] is False
    assert "Connection refused" in data["detail"]
```

---

## Testing the chat endpoint

```python
# tests/test_chat.py
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from saathi.api.main import app
from saathi.api.dependencies import get_graph

class MockGraph:
    async def ainvoke(self, input, config=None):
        msg = input["messages"][0].content
        return {
            "messages": [
                HumanMessage(content=msg),
                AIMessage(content=f"Echo: {msg}", tool_calls=[]),
            ],
            "mode": input.get("mode", "default"),
            "session_id": input["session_id"],
            "context_paths": [],
        }

app.dependency_overrides[get_graph] = lambda: MockGraph()
client = TestClient(app)

def test_chat_returns_reply():
    resp = client.post("/chat", json={"message": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "Echo: Hello"
    assert "session_id" in data
    assert data["tool_calls_made"] == 0

def test_chat_requires_message():
    """Missing 'message' field should return 422."""
    resp = client.post("/chat", json={"mode": "debug"})
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any(e["loc"][-1] == "message" for e in errors)

def test_chat_rejects_invalid_mode():
    """Invalid mode should return 422."""
    resp = client.post("/chat", json={"message": "hi", "mode": "turbo"})
    assert resp.status_code == 422

def test_chat_custom_session_id():
    """Custom session_id should be reflected in the response."""
    resp = client.post("/chat", json={"message": "hello", "session_id": "my-session"})
    assert resp.json()["session_id"] == "my-session"
```

---

## Testing the sessions endpoint

```python
# tests/test_sessions.py
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from langchain_core.messages import AIMessage, HumanMessage
from saathi.api.main import app
from saathi.api.dependencies import get_graph

class MockGraphWithHistory:
    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        if thread_id == "known-session":
            class Snapshot:
                values = {
                    "messages": [
                        HumanMessage(content="Hello"),
                        AIMessage(content="Hi there!"),
                    ],
                    "mode": "default",
                }
            return Snapshot()
        return None  # unknown session

app.dependency_overrides[get_graph] = lambda: MockGraphWithHistory()
client = TestClient(app)

def test_get_history_known_session():
    resp = client.get("/sessions/known-session/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "known-session"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "human"
    assert data["messages"][1]["role"] == "ai"

def test_get_history_unknown_session():
    resp = client.get("/sessions/unknown-xyz/history")
    assert resp.status_code == 404
    assert "unknown-xyz" in resp.json()["detail"]

def test_create_session():
    resp = client.post("/sessions", json={"session_id": "new-session", "mode": "debug"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["session_id"] == "new-session"
    assert data["mode"] == "debug"
    assert data["messages"] == []
```

---

## Async tests with `AsyncClient`

For testing async features (like SSE streaming), use `httpx.AsyncClient` with `ASGITransport`:

```python
import pytest
import json
from httpx import AsyncClient, ASGITransport
from saathi.api.main import app
from saathi.api.dependencies import get_graph

class MockStreamingGraph:
    async def astream(self, input, config=None, stream_mode=None):
        from langchain_core.messages import AIMessageChunk
        tokens = ["Hello", " from", " the", " mock", " agent"]
        for token in tokens:
            yield AIMessageChunk(content=token), {}

app.dependency_overrides[get_graph] = lambda: MockStreamingGraph()

@pytest.mark.asyncio
async def test_chat_stream():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream", json={"message": "Hello"}) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

            collected = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    collected.append(chunk)
                    if chunk["done"]:
                        break

    tokens = [c["delta"] for c in collected if not c["done"]]
    assert "".join(tokens) == "Hello from the mock agent"
    assert collected[-1]["done"] is True
```

---

## Fixtures for clean test isolation

Rather than mutating `app.dependency_overrides` in module scope, use pytest fixtures with cleanup:

```python
import pytest
from fastapi.testclient import TestClient
from saathi.api.main import app
from saathi.api.dependencies import get_graph

@pytest.fixture
def mock_graph():
    return MockGraph()

@pytest.fixture
def client(mock_graph):
    app.dependency_overrides[get_graph] = lambda: mock_graph
    yield TestClient(app)
    app.dependency_overrides.clear()  # cleanup after each test

def test_something(client):
    resp = client.get("/health")
    ...
```

---

## Summary

- `TestClient(app)` runs your ASGI app in-process — no server needed.
- `app.dependency_overrides[get_function] = lambda: mock` replaces any dependency with a mock.
- `AsyncClient(transport=ASGITransport(app=app))` enables async tests, including SSE streaming.
- Test validation errors (422) by sending bad request bodies — you get Pydantic's error format.
- Use pytest fixtures to set up and tear down dependency overrides cleanly between tests.
- `asyncio_mode = "auto"` in `pyproject.toml` removes the need for `@pytest.mark.asyncio` on every async test.

---

*Previous: [Chapter 11 — Error Handling](ch-11-error-handling.md)*  
*Next: [Chapter 13 — Middleware](ch-13-middleware.md)*
