# Chapter 10 — Lifespan — Startup and Shutdown

> **What you'll learn:** why you need startup/shutdown hooks, the old `@app.on_event` pattern, the modern `asynccontextmanager` lifespan, and a line-by-line walkthrough of how Saathi builds its agent graph once at startup.

---

## The problem — resources that must be initialised once

Some resources are too expensive to create on every request and must be initialised once when the server starts:

- Database connection pools
- Loaded ML models
- Compiled LangGraph agent graphs
- External service clients with warmup

You could create these as module-level globals at import time, but that won't work for async resources — you can't `await` at the top level of a module in a regular Python script. The application startup hook is the right place.

---

## The old way — `@app.on_event` (deprecated)

FastAPI originally used event hooks:

```python
@app.on_event("startup")
async def startup():
    app.state.db = await connect_to_database()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()
```

This works but has problems: startup and shutdown are separate functions with no shared scope, the pattern is verbose, and `@app.on_event` was deprecated in FastAPI 0.95 (2023).

> **Experienced note:** If you see tutorials using `@app.on_event`, they're outdated. The lifespan pattern replaced it completely and is the current standard.

---

## The modern way — `asynccontextmanager` lifespan

Python's `contextlib.asynccontextmanager` decorator turns an async generator function into an async context manager. FastAPI uses this as the lifespan:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("Server starting up...")
    # Initialise resources here
    yield                    # ← server runs here
    # --- SHUTDOWN ---
    print("Server shutting down...")
    # Clean up resources here

app = FastAPI(lifespan=lifespan)
```

Everything before `yield` runs once when the server starts (before the first request). Everything after `yield` runs when the server shuts down (after the last request). The `yield` point is where your application lives.

The `app` parameter is the FastAPI application instance — rarely needed, but it lets you attach state to `app.state`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await create_connection_pool()
    yield
    await app.state.db.close()
```

Handlers can then access `app.state.db` via `Request.app.state.db` — though Saathi uses `Depends()` instead (cleaner).

---

## Saathi's lifespan — complete walkthrough

```python
# saathi-langgraph/src/saathi/api/main.py

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from saathi.agent.graph import build_graph, close_graph
from saathi.api import dependencies
from saathi.memory.store import MemoryStore
from saathi.tools import ALL_TOOLS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────
    dependencies._memory_store = MemoryStore()
    dependencies._graph = await build_graph(
        tools=ALL_TOOLS,
        memory_store=dependencies._memory_store,
        db_path=Path(".saathi") / "api_checkpoints.db",
    )
    yield
    # ── SHUTDOWN ─────────────────────────────────────────────
    await close_graph(dependencies._graph)

app = FastAPI(
    title="Saathi API",
    version="1.0.0",
    lifespan=lifespan,
)
```

**Line by line:**

**`dependencies._memory_store = MemoryStore()`** — creates the two-tier JSON memory store (reads from `~/.saathi/memory.json` and `.saathi/memory.json`). This is synchronous — no `await` needed.

**`dependencies._graph = await build_graph(...)`** — this is the expensive async operation. `build_graph` opens an async SQLite connection (via `aiosqlite`), sets up the LangGraph checkpointer, builds the StateGraph, and compiles it. This can't be done at module level because it requires an active event loop.

**`tools=ALL_TOOLS`** — the full list of 15 tools (read_file, run_bash, git_diff, search_web, etc.) is bound to the LLM during compilation. The compiled graph knows which tools are available.

**`db_path=Path(".saathi") / "api_checkpoints.db"`** — a separate SQLite database from the CLI's checkpoints. This prevents the API from interfering with CLI sessions.

**`yield`** — the server is now ready. FastAPI starts accepting requests. This is where the server "lives" — potentially for days or weeks in production.

**`await close_graph(dependencies._graph)`** — on shutdown (CTRL+C, signal, container stop), closes the SQLite connection cleanly. Without this, there may be uncommitted writes or file corruption.

---

## Why module-level globals are set here

You might wonder: why write to `dependencies._graph` from inside `lifespan`? Why not build the graph inside `get_graph()`?

`get_graph()` is a sync function — it can't `await`. And even if it were async, you'd want the graph built *once*, not once-per-request. The lifespan is the right place for async initialisation of shared state.

The `dependencies` module serves as a shared mutable namespace. `lifespan` writes to it at startup; `get_graph()` reads from it on every request. It's a simple, explicit singleton pattern.

---

## Testing with lifespan

When writing tests, you often want to skip the real lifespan (no real Ollama, no real SQLite). FastAPI's `TestClient` does not run the lifespan by default in older patterns, but `httpx.AsyncClient` with `asgi_transport` does:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from saathi.api.main import app

@pytest.fixture
async def client():
    # Override the graph dependency so no Ollama needed
    from saathi.api.dependencies import get_graph
    app.dependency_overrides[get_graph] = lambda: MockGraph()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
```

Chapter 12 covers this in full.

---

## Lifespan for a database connection pool

Here is a production-grade lifespan example using SQLAlchemy async (a common real-world pattern):

```python
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

engine: AsyncEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = create_async_engine("postgresql+asyncpg://user:pass@db/mydb", pool_size=10)
    yield
    await engine.dispose()
```

The shared scope of the `asynccontextmanager` function means `engine` is accessible in both startup and shutdown — a natural place to use module-level state or closures.

---

## Summary

- Resources that must be initialised once (ML models, DB pools, compiled graphs) belong in the lifespan.
- `@asynccontextmanager async def lifespan(app)` — everything before `yield` is startup, everything after is shutdown.
- Pass `lifespan=lifespan` to `FastAPI()`.
- Saathi builds its LangGraph agent graph asynchronously in the lifespan, populates module-level globals in `dependencies.py`, and closes the SQLite connection on shutdown.
- `@app.on_event` is deprecated — use `asynccontextmanager` lifespan instead.

---

*Previous: [Chapter 9 — Streaming Responses and SSE](ch-09-streaming-sse.md)*  
*Next: [Chapter 11 — Error Handling](ch-11-error-handling.md)*
