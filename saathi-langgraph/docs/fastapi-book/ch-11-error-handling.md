# Chapter 11 — Error Handling

> **What you'll learn:** `HTTPException`, the automatic 422 validation error, custom exception handlers, overriding default error responses, and the 404 in `GET /sessions/{id}/history`.

---

## The two kinds of errors in an API

**Client errors (4xx)** — the client sent something wrong: a missing required field, an invalid session ID, unauthorised access. These are expected and should be returned with a clear message.

**Server errors (5xx)** — something went wrong on the server: the database is down, the LLM threw an exception, there's a bug. These should be logged, not exposed in detail to clients.

FastAPI handles the second kind automatically (internal errors become 500 responses). Your job is to handle the first kind — signalling clearly to the client what they did wrong.

---

## `HTTPException` — raising HTTP errors

`HTTPException` is how you signal a client error from anywhere in your handler or dependency:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in database:
        raise HTTPException(status_code=404, detail="Item not found")
    return database[item_id]
```

When you `raise HTTPException`, FastAPI immediately stops processing the request and sends the error response. Your handler code after the raise never runs.

The response body is:

```json
{"detail": "Item not found"}
```

`detail` can be a string, dict, or list — whatever gives the client the most useful information:

```python
raise HTTPException(
    status_code=422,
    detail={
        "error": "Invalid mode",
        "received": "turbo",
        "allowed": ["default", "explain", "refactor", "debug"],
    }
)
```

---

## The 404 in `sessions.py` — a real example

```python
@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, graph: GraphDep) -> SessionHistoryResponse:
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)

    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    ...
```

`graph.aget_state(config)` returns `None` (or a snapshot with empty values) when the thread ID has no checkpoint history. There's no such session. The correct response is 404.

The `detail` includes the `session_id` — this is good practice. The client knows exactly which ID was invalid, which is especially helpful when debugging.

---

## 422 — automatic validation errors

When a request body fails Pydantic validation, FastAPI returns 422 automatically. You do not write this code — it's built in.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"mode": "turbo"}'   # missing "message", invalid mode
```

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "message"],
      "msg": "Field required",
      "input": {"mode": "turbo"},
      "url": "https://errors.pydantic.dev/2.7/v/missing"
    },
    {
      "type": "literal_error",
      "loc": ["body", "mode"],
      "msg": "Input should be 'default', 'explain', 'refactor' or 'debug'",
      "input": "turbo",
      "url": "https://errors.pydantic.dev/2.7/v/literal_error"
    }
  ]
}
```

Each error in the `detail` list tells you:
- `type` — what kind of error
- `loc` — where in the request it came from (`body`, `path`, `query`)
- `msg` — human-readable description
- `input` — what was actually received

This is Pydantic v2's standard error format. You get this for free on every endpoint.

---

## Custom exception handlers

For application-wide errors that don't fit `HTTPException`, register a custom exception handler:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class OllamaUnavailableError(Exception):
    pass

@app.exception_handler(OllamaUnavailableError)
async def ollama_error_handler(request: Request, exc: OllamaUnavailableError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "LLM unavailable",
            "detail": str(exc),
            "suggestion": "Check that Ollama is running: ollama serve",
        },
    )

# Now anywhere in the app:
raise OllamaUnavailableError("Connection refused to localhost:11434")
# → 503 with the structured error body
```

This is better than raising `HTTPException(503, ...)` everywhere — the error handling logic is centralised, and you can add logging, metrics, or alerting in one place.

---

## Overriding the default 422 response

Sometimes you want a different format for validation errors. Override the built-in handler:

```python
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    return JSONResponse(
        status_code=422,
        content={
            "message": "Request validation failed",
            "errors": [
                {"field": " → ".join(str(loc) for loc in e["loc"]), "reason": e["msg"]}
                for e in errors
            ],
        },
    )
```

Result:

```json
{
  "message": "Request validation failed",
  "errors": [
    {"field": "body → message", "reason": "Field required"},
    {"field": "body → mode", "reason": "Input should be 'default', 'explain', 'refactor' or 'debug'"}
  ]
}
```

> **Experienced note:** DRF has very similar custom error handling via `EXCEPTION_HANDLER` in settings. Flask uses `@app.errorhandler`. FastAPI's `@app.exception_handler` works the same way.

---

## Adding response error documentation

By default, only the success response is documented. You can document error responses too:

```python
@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    responses={
        404: {"description": "Session not found", "content": {"application/json": {"example": {"detail": "Session 'abc123' not found."}}}},
        500: {"description": "Internal server error"},
    },
)
async def get_session_history(...):
    ...
```

These appear in the Swagger UI under the endpoint's response section.

---

## Unhandled exceptions — 500

If your handler raises an exception you didn't catch, FastAPI catches it and returns:

```json
{"detail": "Internal Server Error"}
```

In development with `--reload`, you'll also see the traceback in the uvicorn console. In production, the traceback is hidden from the client (correctly — you don't want stack traces exposed publicly).

To add logging for unexpected errors:

```python
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

---

## Summary

- `raise HTTPException(status_code=..., detail=...)` signals any client error and stops handler execution.
- 422 Unprocessable Entity is returned automatically by FastAPI when Pydantic validation fails — you write no code.
- `@app.exception_handler(ExceptionType)` registers a handler for specific exception types — use for application-wide errors.
- Override `RequestValidationError` handler to customise the 422 response format.
- Document error responses with `responses={404: ..., 500: ...}` on the route decorator.

---

*Previous: [Chapter 10 — Lifespan](ch-10-lifespan.md)*  
*Next: [Chapter 12 — Testing FastAPI Apps](ch-12-testing.md)*
