# Chapter 8 — Routers — Organising a Real Project

> **What you'll learn:** why putting all routes in one file doesn't scale, how `APIRouter` splits routes across files, the `prefix` and `tags` options, `include_router`, and how the Saathi project is structured.

---

## The problem with one big file

When you're building your first FastAPI app, putting everything in `main.py` is fine:

```python
# main.py — works, but gets unwieldy fast

app = FastAPI()

@app.get("/health")
def health(): ...

@app.get("/model/info")
def model_info(): ...

@app.post("/chat")
async def chat(): ...

@app.post("/chat/stream")
async def chat_stream(): ...

@app.post("/sessions")
async def create_session(): ...

@app.get("/sessions/{session_id}/history")
async def get_session_history(): ...
```

Six endpoints is manageable. Twenty isn't. A hundred is a nightmare. You can't find anything, imports pile up at the top of the file, and any change risks breaking something unrelated.

`APIRouter` solves this by letting you define groups of related routes in separate files and then register them all in `main.py`.

---

## `APIRouter` — the building block

`APIRouter` works exactly like `FastAPI()` for route decoration, but it's not an application — it's a group of routes that gets attached to one later:

```python
# routes/health.py
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

`tags=["health"]` groups this endpoint under the "health" section in the Swagger UI. Without tags, all endpoints appear in a single flat list.

---

## Registering routers with `include_router`

In `main.py`, you pull all the routers together:

```python
# main.py
from fastapi import FastAPI
from saathi.api.routes import chat, health, model, sessions

app = FastAPI(title="Saathi API", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(model.router)
app.include_router(chat.router)
app.include_router(sessions.router)
```

`include_router` adds every route from the router into the application. The routes behave identically to if they'd been defined on `app` directly.

---

## The `prefix` option

`prefix` prepends a path segment to all routes in a router. Instead of writing `/sessions` in every path:

```python
# Without prefix — repetitive
@router.post("/sessions")
@router.get("/sessions/{session_id}/history")

# With prefix — cleaner
router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("")              # → POST /sessions
@router.get("/{session_id}/history")   # → GET /sessions/{session_id}/history
```

The Saathi sessions router uses a prefix:

```python
# routes/sessions.py
router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("", response_model=SessionHistoryResponse, status_code=201)
async def create_session(...): ...

@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(...): ...
```

The empty string `""` on `@router.post("")` means "the root of this router's prefix" — i.e., `POST /sessions`. The `/{session_id}/history` becomes `GET /sessions/{session_id}/history`.

> **Experienced note:** Express.js `Router` works the same way: `router.get('/:id', handler)` then `app.use('/sessions', router)`. FastAPI's equivalent is `include_router(router, prefix="/sessions")` or prefix on the router itself.

---

## The full Saathi router structure

```
main.py
├── include_router(health.router)     → GET /health
├── include_router(model.router)      → GET /model/info
├── include_router(chat.router)       → POST /chat
│                                        POST /chat/stream
└── include_router(sessions.router)   → POST /sessions
                                         GET /sessions/{session_id}/history
```

Each router file is self-contained: it imports only what it needs (schemas, dependencies), defines its routes, and exports `router`. `main.py` knows nothing about the implementation details.

---

## `tags` — organising the Swagger UI

Tags group endpoints in the Swagger UI under named sections:

```python
router = APIRouter(tags=["chat"])
```

Without tags, `/health`, `/model/info`, `/chat`, `/chat/stream`, `/sessions` all appear in a single "default" section. With tags:

```
▼ health
    GET /health

▼ model
    GET /model/info

▼ chat
    POST /chat
    POST /chat/stream

▼ sessions
    POST /sessions
    GET  /sessions/{session_id}/history
```

Tags can also carry descriptions:

```python
app = FastAPI(openapi_tags=[
    {"name": "chat", "description": "Send messages to the Saathi agent"},
    {"name": "sessions", "description": "Manage conversation sessions"},
])
```

---

## Router-level dependencies

You can attach a dependency to an entire router, and it will run for every route in that router:

```python
from fastapi import Depends

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)

@router.get("/stats")    # auth runs automatically
def get_stats(): ...

@router.delete("/users/{id}")  # auth runs automatically
def delete_user(): ...
```

This is cleaner than adding `Depends(require_admin_token)` to every handler individually.

---

## Adding a prefix when registering

You can also supply the prefix at `include_router` time instead of on the router itself:

```python
app.include_router(sessions.router, prefix="/api/v1")
# → POST /api/v1/sessions
# → GET  /api/v1/sessions/{session_id}/history
```

This is useful for versioning — you can add a `/v2` prefix when you create the v2 router without touching the route files themselves.

---

## Sub-applications with `mount`

For large projects, you might want entirely separate FastAPI applications with separate `/docs`. Use `app.mount`:

```python
from fastapi import FastAPI

main_app = FastAPI()
admin_app = FastAPI()

@admin_app.get("/stats")
def admin_stats(): ...

main_app.mount("/admin", admin_app)
# Admin app is now accessible at /admin/stats
# Admin app has its own /admin/docs
```

This is heavier than `include_router` — use it when the sub-application has truly different concerns (different middleware, different auth, separate docs).

---

## Summary

- `APIRouter` groups related routes into a file and lets them be registered together.
- `prefix` prepends a path segment to all routes in a router.
- `tags` groups endpoints in the Swagger UI.
- `include_router` attaches a router to the application (optionally with an additional prefix).
- Router-level `dependencies=[]` applies a dependency to every route in the router.
- The Saathi project has one router per domain area: health, model, chat, sessions.

---

*Previous: [Chapter 7 — Async Python in FastAPI](ch-07-async-python.md)*  
*Next: [Chapter 9 — Streaming Responses and SSE](ch-09-streaming-sse.md)*
