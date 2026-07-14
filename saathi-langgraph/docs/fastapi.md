# Saathi FastAPI — REST API for the LangGraph Agent

The `saathi.api` package wraps the Saathi LangGraph agent in a fully async FastAPI server, letting you interact with the agent over HTTP instead of (or alongside) the CLI.

---

## Prerequisites

- Ollama running locally: `ollama serve` (default `http://localhost:11434`)
- The model pulled: `ollama pull gemma4:12b`
- Dependencies installed in your venv (see [Setup](#setup))

---

## Setup

```bash
# From the repo root — installs saathi-langgraph and its deps (including FastAPI + uvicorn)
cd "c:/development/Python/Neural Network"
.venv/Scripts/pip install -e saathi-langgraph
```

---

## Running the server

### Option A — uvicorn directly (recommended for development)

```bash
cd "c:/development/Python/Neural Network"

# Windows PowerShell
$env:PYTHONPATH = "saathi-langgraph/src"
.venv/Scripts/python.exe -m uvicorn saathi.api.main:app --reload --port 8000

# Git Bash / WSL
PYTHONPATH=saathi-langgraph/src .venv/Scripts/python.exe -m uvicorn saathi.api.main:app --reload --port 8000
```

`--reload` watches for file changes and restarts automatically — great while learning.

### Option B — installed entry point

```bash
# After `pip install -e saathi-langgraph` with the shared venv active:
saathi-api
# Starts on http://0.0.0.0:8000 (no auto-reload)
```

### Option C — module run

```bash
PYTHONPATH=saathi-langgraph/src .venv/Scripts/python.exe -m saathi.api.server
```

---

## Web UI

Open **`http://localhost:8000`** in your browser to get a full dark-themed chat interface — no extra setup needed.

```uri
http://localhost:8000        ← chat UI (served from api/static/index.html)
http://localhost:8000/docs   ← Swagger UI (API reference)
http://localhost:8000/redoc  ← ReDoc
```

### What the UI provides

| Feature | Detail |
| --- | --- |
| **Streaming chat** | Tokens stream in real-time via `/chat/stream` (SSE) |
| **Mode selector** | Switch between Default / Explain / Refactor / Debug per message |
| **Context paths** | Optionally scope the agent to specific files or directories |
| **Session management** | Multiple named sessions in the sidebar, each with its own thread ID |
| **Health indicator** | Live coloured dot in the header — polls `/health` on load |
| **Model info panel** | Shows model name, temperature, context window pulled from `/model/info` |
| **Markdown rendering** | Code blocks, inline code, bold, and headings rendered inside bubbles |

### How the UI connects to FastAPI

The UI is a single HTML file (`api/static/index.html`) mounted on `StaticFiles` and served at `/`. It talks to the same API endpoints you'd call with `curl`:

```endpoints
GET  /health          → status dot colour
GET  /model/info      → sidebar model panel
POST /chat/stream     → streaming SSE tokens into the chat bubble
```

The `StaticFiles` mount and the redirect of `/` to `index.html` live in `api/main.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_STATIC / "index.html")
```

> **FastAPI concept:** `StaticFiles` is a sub-application mounted at a path prefix — any request to `/static/*` is handled entirely by Starlette's file server, never reaching your route handlers.

---

## Interactive API docs

Once the server is running, open your browser:

| URL | What you get |
| --- | --- |
| `http://localhost:8000` | **Chat UI** — full graphical interface |
| `http://localhost:8000/docs` | **Swagger UI** — try every endpoint live |
| `http://localhost:8000/redoc` | ReDoc — cleaner read-only reference |
| `http://localhost:8000/openapi.json` | Raw OpenAPI schema |

> **Tip:** Use `/docs` to explore and experiment before writing any client code.

---

## Endpoints

### `GET /health`

Check that the server is up and Ollama is reachable.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "ollama_reachable": true,
  "model": "gemma4:12b",
  "detail": null
}
```

If Ollama is not running, `status` will be `"degraded"` and `detail` will explain why.

---

### `GET /model/info`

See the active LLM configuration (reads from `SAATHI_*` env vars / `.env`).

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model": "gemma4:12b",
  "base_url": "http://localhost:11434",
  "temperature": 0.1,
  "context_window": 32768,
  "max_tokens": 4096,
  "max_parallel_tools": 8
}
```

---

### `POST /chat` — full response

Send a message and wait for the complete reply. The agent runs its full ReAct loop (including any tool calls) before responding.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What Python files are in this project?",
       "mode": "default"
     }'
```

**Request body:**

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `message` | string | yes | — | Your message to the agent |
| `session_id` | string | no | auto UUID | Reuse to continue a conversation |
| `mode` | string | no | `"default"` | `default` · `explain` · `refactor` · `debug` |
| `context_paths` | list[string] | no | `[]` | File/dir paths to scope the agent to |

**Response:**

```json
{
  "session_id": "3f8a2b1c-...",
  "reply": "The project contains the following Python files: ...",
  "tool_calls_made": 2
}
```

**Continuing a conversation** — pass the same `session_id`:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Now show me the contents of main.py",
       "session_id": "3f8a2b1c-..."
     }'
```

---

### `POST /chat/stream` — streaming (Server-Sent Events)

Streams the agent's reply token-by-token. Useful for building a UI or watching long answers appear in real time.

```bash
curl -N -X POST http://localhost:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain the LangGraph agent architecture"}'
```

Each event is a line like:

```text
data: {"session_id": "abc...", "delta": "The ", "done": false}
data: {"session_id": "abc...", "delta": "agent ", "done": false}
...
data: {"session_id": "abc...", "delta": "", "done": true}
```

The `done: true` event signals the end of the stream. Connect from Python:

```python
import httpx

with httpx.Client(timeout=None) as client:
    with client.stream(
        "POST",
        "http://localhost:8000/chat/stream",
        json={"message": "Refactor this function", "mode": "refactor"},
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                import json
                chunk = json.loads(line[6:])
                print(chunk["delta"], end="", flush=True)
                if chunk["done"]:
                    break
```

---

### `POST /sessions` — create a session

Pre-register a session with a custom ID and mode before sending any messages.

```bash
curl -X POST http://localhost:8000/sessions \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-debug-session", "mode": "debug"}'
```

```json
{
  "session_id": "my-debug-session",
  "mode": "debug",
  "messages": []
}
```

You can skip this and just pass `session_id` directly to `/chat` — the session is created automatically on first use.

---

### `GET /sessions/{session_id}/history`

Retrieve the full conversation history for a session from the SQLite checkpoint store.

```bash
curl http://localhost:8000/sessions/my-debug-session/history
```

```json
{
  "session_id": "my-debug-session",
  "mode": "debug",
  "messages": [
    { "role": "human", "content": "Why is this function slow?" },
    { "role": "ai",    "content": "Looking at the code, the issue is..." }
  ]
}
```

Returns `404` if the session has no history yet.

---

## Configuration

All settings are read from environment variables (prefix `SAATHI_`) or a `.env` file in your working directory.

```env
# .env
SAATHI_OLLAMA_MODEL=gemma4:12b
SAATHI_OLLAMA_BASE_URL=http://localhost:11434
SAATHI_TEMPERATURE=0.1
SAATHI_MAX_TOKENS=4096
SAATHI_CONTEXT_WINDOW=32768
```

Change `SAATHI_OLLAMA_MODEL` to switch models without touching code:

```bash
SAATHI_OLLAMA_MODEL=llama3.2:3b python -m uvicorn saathi.api.main:app --port 8000
```

---

## FastAPI concepts in this project

If you're learning FastAPI, here's where each concept appears:

| Concept | File | What to look at |
| --- | --- | --- |
| **`FastAPI()` app** | `api/main.py` | App creation, `include_router` |
| **Lifespan** (startup/shutdown) | `api/main.py` | `@asynccontextmanager` builds the graph once |
| **`APIRouter`** | `api/routes/*.py` | Splitting routes across files |
| **Pydantic models** | `api/schemas.py` | Request/response validation + auto-docs |
| **`Depends()`** | `api/dependencies.py` | Injecting shared singletons into handlers |
| **Path parameters** | `api/routes/sessions.py` | `/{session_id}/history` |
| **`response_model`** | every endpoint | Auto-serialises + documents return type |
| **`StreamingResponse`** | `api/routes/chat.py` | SSE token streaming |
| **`StaticFiles`** | `api/main.py` | Mount a directory to serve static files (the chat UI) |
| **`FileResponse`** | `api/main.py` | Return a file directly as an HTTP response |
| **Async handlers** | all routes | `async def` + `await graph.ainvoke()` |

---

## File layout

```folder
src/saathi/api/
├── main.py          # FastAPI app + lifespan + StaticFiles mount
├── server.py        # saathi-api CLI entry point
├── dependencies.py  # Shared graph/memory singletons via Depends()
├── schemas.py       # Pydantic request & response models
├── static/
│   └── index.html   # Self-contained chat UI (served at /)
└── routes/
    ├── health.py    # GET  /health
    ├── model.py     # GET  /model/info
    ├── chat.py      # POST /chat  +  POST /chat/stream
    └── sessions.py  # POST /sessions  +  GET /sessions/{id}/history
```
