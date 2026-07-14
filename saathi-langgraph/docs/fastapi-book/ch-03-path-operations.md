# Chapter 3 — Path Operations

> **What you'll learn:** HTTP methods and when to use each, path parameters, query parameters, and status codes — all grounded in the Saathi health and model endpoints.

---

## What is a path operation?

FastAPI calls a route handler a **path operation** — it is the combination of an HTTP method (the *operation*) and a URL path. `@app.get("/health")` means "when someone makes a GET request to /health, run this function."

The main decorators are:

```python
@app.get("/items")        # Read data
@app.post("/items")       # Create data
@app.put("/items/{id}")   # Replace data entirely
@app.patch("/items/{id}") # Update part of the data
@app.delete("/items/{id}")# Delete data
```

> **Beginner note:** These HTTP methods are a convention, not a technical enforcement. GET could technically create data. But following REST conventions makes your API predictable for everyone who uses it.

---

## HTTP methods — when to use each

**GET** — retrieve data. No body. Safe (no side effects) and idempotent (calling it twice gives the same result). Use for reading: `/health`, `/model/info`, `/sessions/{id}/history`.

**POST** — create something new or trigger an action. Has a body. Not idempotent — calling it twice creates two things (or triggers the action twice). Use for creating sessions, sending chat messages.

**PUT** — replace a resource entirely. Idempotent — calling it twice results in the same state.

**PATCH** — partially update a resource. Use when you only want to update specific fields.

**DELETE** — remove a resource. Idempotent.

In Saathi, there are only GET and POST endpoints. There is no concept of "update a message" or "delete a session" in a chat API, so PUT/PATCH/DELETE are not used.

---

## Path parameters

A path parameter is a variable segment inside the URL path. You declare it with curly braces in the path and as a function parameter with a matching name:

```python
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, graph: GraphDep):
    ...
```

The `: str` type annotation tells FastAPI to:
1. Extract the value from the URL
2. Validate it is a string (trivially true for path params)
3. Pass it to the function

Type annotations matter more for numeric types:

```python
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}
```

```bash
curl http://localhost:8000/items/42     # works → {"item_id": 42}
curl http://localhost:8000/items/abc    # 422 error → "value is not a valid integer"
```

FastAPI validates the type before calling your handler. You never write `try: int(item_id) except ValueError: ...`.

> **Experienced note:** In Flask you'd write `@app.route("/items/<int:item_id>")`. FastAPI infers the same from the Python type hint — no converter syntax needed.

### Path parameter in Saathi

```python
# routes/sessions.py
@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, graph: GraphDep) -> SessionHistoryResponse:
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    ...
```

The `session_id` in `/{session_id}/history` maps directly to the `session_id: str` parameter. FastAPI passes the extracted string straight in — no parsing code needed.

---

## Query parameters

Query parameters are the key-value pairs after `?` in a URL: `/items?skip=0&limit=10`.

In FastAPI, any function parameter that is **not** a path parameter and **not** a body is automatically treated as a query parameter:

```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

```bash
curl "http://localhost:8000/items?skip=20&limit=5"
# {"skip": 20, "limit": 5}

curl "http://localhost:8000/items"
# {"skip": 0, "limit": 10}   ← defaults used
```

### Optional query parameters

Use `None` as the default to make a query parameter optional:

```python
from typing import Optional

@app.get("/search")
def search(query: str, category: str | None = None):
    if category:
        return {"query": query, "category": category}
    return {"query": query}
```

```bash
curl "http://localhost:8000/search?query=fastapi"
curl "http://localhost:8000/search?query=fastapi&category=docs"
```

### Query params in Saathi

The Saathi API doesn't use query params much — the main routes use POST bodies for all input. But query params are the right choice for filtering/pagination. If Saathi had a "list all sessions" endpoint, it would look like:

```python
@router.get("", response_model=list[SessionSummary])
async def list_sessions(skip: int = 0, limit: int = 20, graph: GraphDep = ...):
    ...
```

---

## The health endpoint — a complete walkthrough

```python
# routes/health.py
import httpx
from fastapi import APIRouter
from saathi.api.schemas import HealthResponse
from saathi.config import settings

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        reachable = resp.status_code == 200
        detail = None if reachable else f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        reachable = False
        detail = str(exc)

    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        model=settings.ollama_model,
        detail=detail,
    )
```

Breaking this down:

- `@router.get("/health")` — GET method, path `/health` (the prefix `/` comes from the parent router in `main.py`)
- `response_model=HealthResponse` — FastAPI will validate the return value against this Pydantic model and use it to generate the response schema in docs
- `async def` — this handler makes an outbound HTTP call (I/O), so it's correctly declared async
- `httpx.AsyncClient` — the async HTTP client; using the sync `requests` library here would block the event loop (Chapter 7 covers this)
- Returns a `HealthResponse` instance — FastAPI serialises it to JSON

---

## The model/info endpoint

```python
# routes/model.py
from fastapi import APIRouter
from saathi.api.schemas import ModelInfoResponse
from saathi.config import settings

router = APIRouter(tags=["model"])

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        context_window=settings.context_window,
        max_tokens=settings.max_tokens,
        max_parallel_tools=settings.max_parallel_tools,
    )
```

This is the simplest possible real endpoint: no parameters, no I/O, just reads configuration and returns a structured response. This is a useful pattern for debugging — hit `/model/info` to confirm the server is using the right model and settings.

---

## Status codes

HTTP responses include a numeric status code that signals whether the request succeeded:

| Code | Meaning | When to use |
|---|---|---|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST that creates a resource |
| 204 | No Content | Successful DELETE (no body) |
| 400 | Bad Request | Client sent malformed data |
| 404 | Not Found | Resource doesn't exist |
| 422 | Unprocessable Entity | Pydantic validation failed |
| 500 | Internal Server Error | Unexpected server-side error |

FastAPI defaults to 200 for all successful responses. To use a different code, pass `status_code` to the decorator:

```python
@router.post("", response_model=SessionHistoryResponse, status_code=201)
async def create_session(body: SessionCreateRequest, graph: GraphDep):
    ...
```

Notice `POST /sessions` returns 201 — it creates a new session resource. The 404 for unknown sessions comes from `raise HTTPException(status_code=404, ...)` — covered in Chapter 11.

---

## Documenting your endpoints

FastAPI uses your function's docstring as the endpoint description in the docs:

```python
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check whether the API is running and Ollama is reachable."""
    ...
```

This text appears in both Swagger UI and ReDoc. It's worth writing — it costs one line and makes your API much more usable by others.

You can also add summaries and descriptions directly to the decorator:

```python
@app.get(
    "/health",
    summary="Health check",
    description="Pings Ollama and reports whether the model server is reachable.",
    response_description="Service status and Ollama connectivity",
)
```

---

## Summary

- Path operations are the combination of an HTTP method decorator and a URL path.
- Path parameters `{name}` are extracted and type-validated automatically.
- Query parameters are any function parameters that aren't path params or body — they get `?key=value` in the URL.
- Default values make parameters optional.
- `status_code=` on the decorator controls the HTTP status for successful responses.
- Docstrings become endpoint descriptions in the auto-generated docs.

---

*Previous: [Chapter 2 — How FastAPI Works Under the Hood](ch-02-how-it-works.md)*  
*Next: [Chapter 4 — Pydantic Models](ch-04-pydantic-models.md)*
