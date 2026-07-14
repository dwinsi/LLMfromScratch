# Chapter 2 — How FastAPI Works Under the Hood

> **What you'll learn:** ASGI vs WSGI, how Starlette and Pydantic fit in, and what happens to an HTTP request from the moment it arrives to the moment your handler runs.

---

## The two eras of Python web frameworks

To understand FastAPI's architecture you need to know why it was built differently from Flask and Django.

### WSGI — the old way

Flask and Django are built on **WSGI** (Web Server Gateway Interface), a standard from 2003. WSGI defines a simple contract: a web server calls your application as a Python function, passes the request as arguments, and your application returns a response.

```
Browser → nginx → Gunicorn → Flask app
```

The critical limitation: each WSGI request occupies one OS thread from start to finish. When your Flask handler calls a database, that thread sits blocked — doing nothing — until the database responds. Under high load you need many threads, which means high memory use and expensive context switching.

> **Experienced note:** This is why "concurrent connections" benchmarks favour async frameworks so dramatically. It's not that sync Python is slow at computation — it's that threads waiting on I/O waste resources.

### ASGI — the new way

**ASGI** (Asynchronous Server Gateway Interface) was designed as WSGI's async successor. Instead of a blocking function call, ASGI uses Python's `asyncio` coroutine model. While one request is waiting on a database response, the event loop can handle dozens of other requests on the same thread.

```
Browser → nginx → Uvicorn (ASGI) → FastAPI app
```

FastAPI is an ASGI application. Uvicorn is an ASGI server — it handles the network layer and calls your FastAPI app asynchronously.

---

## FastAPI's three layers

FastAPI is not built from scratch. It sits on top of two existing libraries:

```
┌─────────────────────────────────────────┐
│               FastAPI                   │  ← routing, DI, docs generation
├─────────────────────────────────────────┤
│              Starlette                  │  ← ASGI toolkit, request/response
├─────────────────────────────────────────┤
│               Pydantic                  │  ← data validation and serialisation
└─────────────────────────────────────────┘
```

### Starlette

**Starlette** is a lightweight ASGI framework/toolkit. It provides the primitives: `Request`, `Response`, routing, middleware, background tasks, WebSocket support, and static files. FastAPI is literally a subclass of Starlette's `Starlette` application — `FastAPI` extends it with type hints, automatic validation, and docs generation.

You will rarely interact with Starlette directly, but it explains why FastAPI can do things like `StreamingResponse` and middleware without reinventing them.

### Pydantic

**Pydantic** is a data validation library that uses Python type hints to define data shapes. FastAPI uses Pydantic for two things:
1. **Deserialising** the incoming request body from JSON into a typed Python object.
2. **Serialising** your return value back to JSON for the response.

When you define a `class ChatRequest(BaseModel)`, FastAPI uses Pydantic to validate that every incoming request body matches that shape before your handler ever runs.

---

## The request lifecycle — step by step

Here is exactly what happens when a client sends `POST /chat` to the Saathi API:

```
1. TCP connection arrives at the OS
2. Uvicorn accepts it and reads the raw HTTP bytes
3. Uvicorn parses HTTP/1.1 or HTTP/2 framing
4. Uvicorn calls FastAPI as an ASGI app with the request scope
5. FastAPI matches "/chat" + POST method → finds the chat() handler
6. FastAPI reads the request body
7. Pydantic validates the body against ChatRequest schema
   → if invalid: returns 422 Unprocessable Entity immediately
   → if valid: constructs a ChatRequest instance
8. FastAPI resolves dependencies (Depends(get_graph) → returns the graph)
9. FastAPI calls: await chat(req=<ChatRequest>, graph=<CompiledGraph>)
10. Your handler runs: await graph.ainvoke(...)
11. Handler returns: ChatResponse(session_id=..., reply=..., ...)
12. FastAPI serialises ChatResponse to JSON via Pydantic
13. FastAPI constructs an HTTP 200 response with Content-Type: application/json
14. Uvicorn sends the bytes back over the TCP connection
```

Steps 7, 8, 11, and 12 are things FastAPI does for you automatically. You just write steps 9 and 10.

---

## The OpenAPI schema — where docs come from

FastAPI inspects your code at startup — specifically the type annotations on your route handler functions and the Pydantic models they reference — and builds an **OpenAPI 3.x** specification. This is a standard JSON document that describes your entire API: every endpoint, every request body shape, every response shape.

```bash
curl http://localhost:8000/openapi.json
```

You'll get back a JSON document listing all your paths, their parameters, their request/response schemas. Swagger UI and ReDoc are just JavaScript apps that render this JSON into a human-friendly interface.

The key insight: **you never maintain the docs manually**. They are always in sync with your code because they are derived from your code.

---

## Why type hints are central

In regular Python, type hints are optional annotations that tools can use but Python itself ignores at runtime:

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

FastAPI changes this: it actively reads your type hints at runtime and uses them to:
- Extract and validate path/query/body parameters
- Generate JSON schema for request/response models
- Power editor autocompletion end-to-end

This is why FastAPI functions look slightly more annotated than typical Python — each annotation is doing real work.

---

## Synchronous vs asynchronous handlers

FastAPI supports both `def` and `async def` handlers:

```python
# synchronous — FastAPI runs this in a thread pool
@app.get("/sync")
def sync_handler():
    return {"mode": "sync"}

# asynchronous — FastAPI awaits this on the event loop
@app.get("/async")
async def async_handler():
    return {"mode": "async"}
```

For sync handlers, FastAPI uses a thread pool to avoid blocking the event loop. For async handlers, it `await`s them directly. Chapter 7 covers when to use each and why it matters.

---

## How Saathi uses all of this

The Saathi API is a good illustration of the full stack in action:

- **Uvicorn** runs the ASGI server: `uvicorn saathi.api.main:app`
- **Starlette** (via FastAPI) handles `StreamingResponse` for the SSE streaming endpoint
- **Pydantic** validates every `ChatRequest` body and serialises every `ChatResponse`
- **FastAPI's DI system** injects the compiled LangGraph agent (the `graph` object) into every route handler that needs it, without those handlers having to know how to build the graph

The agent itself — `await graph.ainvoke(...)` — is an async call. If it were synchronous, it would block the event loop while the LLM generates text, making the server unresponsive to other requests. By being async throughout, Saathi can handle multiple concurrent chat requests without threads piling up.

---

## Summary

- **ASGI** is the async-capable successor to WSGI. FastAPI runs on ASGI via Uvicorn.
- FastAPI = **Starlette** (routing, HTTP primitives) + **Pydantic** (validation) + **type hints** (schema inference).
- Every incoming request is validated by Pydantic before reaching your handler.
- The OpenAPI spec is auto-generated from your type hints — no separate docs file needed.
- Both `def` and `async def` handlers are supported; async handlers run on the event loop directly.

---

*Previous: [Chapter 1 — Your First FastAPI App](ch-01-first-app.md)*  
*Next: [Chapter 3 — Path Operations](ch-03-path-operations.md)*
