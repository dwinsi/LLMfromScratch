# Chapter 13 — Middleware

> **What you'll learn:** what middleware is, how `app.add_middleware` works, CORS, GZip, request logging, and when to reach for middleware vs a `Depends()` dependency.

---

## What middleware is

Middleware wraps every request/response that passes through your application. It sits between the ASGI server (uvicorn) and your route handlers:

```
Client → Uvicorn → [Middleware A] → [Middleware B] → Route handler
                 ← [Middleware A] ← [Middleware B] ←
```

On the way in, each middleware can inspect or modify the request. On the way out, it can inspect or modify the response. This makes middleware the right place for cross-cutting concerns that apply to every request: logging, timing, compression, security headers, CORS.

> **Experienced note:** Flask uses `@app.before_request` / `@app.after_request`. Django uses `MIDDLEWARE` in settings. FastAPI uses Starlette's ASGI middleware, which is more composable and type-safe.

---

## `app.add_middleware` — the API

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

app.add_middleware(SomeMiddleware, option1="value", option2=True)
```

Middleware is added in reverse order — the last one added wraps the outermost layer. If you add A then B, the execution order is B → A → handler → A → B.

---

## CORS middleware

**CORS (Cross-Origin Resource Sharing)** controls which web origins are allowed to call your API from a browser. Without CORS headers, a web page at `https://myapp.com` cannot call an API at `https://api.myapp.com` — the browser blocks it.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

| Option | What it controls |
|---|---|
| `allow_origins` | Which origins are permitted. `["*"]` allows all. |
| `allow_credentials` | Whether cookies/auth headers can be sent. |
| `allow_methods` | Allowed HTTP methods (`GET`, `POST`, etc.) |
| `allow_headers` | Allowed request headers |

**For Saathi in production**, you'd restrict to the specific frontend origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # Vite dev server
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

> **Beginner note:** CORS is only enforced by browsers. `curl` and server-to-server calls ignore CORS entirely. If your API is only called by curl or Python code, you don't need CORS at all.

---

## GZip middleware

Compresses responses larger than a threshold. Useful when clients can handle gzip (all modern browsers and most HTTP clients can):

```python
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

`minimum_size=1000` means only compress responses larger than 1000 bytes. For Saathi, a long chat response (several paragraphs) could compress to 30–40% of its original size.

---

## Custom middleware — request logging

Writing a custom middleware with `BaseHTTPMiddleware` gives you full control over the request/response cycle:

```python
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()

        # ── before handler ──────────────────────────────
        logger.info(f"→ {request.method} {request.url.path}")

        response = await call_next(request)   # ← run the handler

        # ── after handler ───────────────────────────────
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            f"← {request.method} {request.url.path} "
            f"{response.status_code} ({elapsed:.1f}ms)"
        )
        return response

app.add_middleware(RequestLoggingMiddleware)
```

`call_next(request)` runs all the remaining middleware and the route handler, then returns the response. You can do work before and after that call.

This would produce log output like:

```
INFO → POST /chat
INFO ← POST /chat 200 (3241.7ms)
INFO → GET /health
INFO ← GET /health 200 (12.3ms)
```

---

## Adding headers to every response

A common use case: add security headers or request ID headers to every response:

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

---

## Trusted hosts middleware

Prevent HTTP Host header attacks — reject requests with unexpected `Host` headers:

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.myapp.com", "localhost", "127.0.0.1"],
)
```

---

## Middleware vs `Depends()` — when to use each

This is a common question. Here's the decision rule:

| Use **middleware** when... | Use **`Depends()`** when... |
|---|---|
| The concern applies to ALL routes | The concern applies to specific routes or routers |
| You need access to request/response headers/body | You need to inject a value INTO the handler |
| You need to modify the response (add headers, compress) | You need to run auth/validation before specific handlers |
| Logging, timing, CORS, compression, security headers | Auth, rate limiting per-user, database session per-request |

A rule of thumb: **middleware wraps, dependencies inject**.

Auth is often implemented as a `Depends()` rather than middleware in FastAPI, because different routes have different auth requirements (public vs admin vs API key). Middleware would need to replicate routing logic. Dependency injection is cleaner.

---

## Middleware and streaming responses

Be careful with `BaseHTTPMiddleware` and streaming responses — the middleware pattern buffers the response body to give you access to it. For SSE streaming (`POST /chat/stream`), this is a problem: the middleware can't hand the body to `call_next` while it's still streaming.

The solution is to write a pure ASGI middleware instead of using `BaseHTTPMiddleware`, or to skip middleware entirely for streaming routes. In practice, the logging and CORS middleware don't buffer the body — they only touch headers — so they work fine with SSE.

---

## Summary

- Middleware wraps every request/response — use it for cross-cutting concerns that apply to all routes.
- `CORSMiddleware` enables browsers to call your API from different origins.
- `GZipMiddleware` compresses large responses.
- Custom `BaseHTTPMiddleware` subclasses give full control over request/response.
- Use `Depends()` instead of middleware when the concern only applies to specific routes.
- Be careful with `BaseHTTPMiddleware` and streaming responses — test explicitly.

---

*Previous: [Chapter 12 — Testing FastAPI Apps](ch-12-testing.md)*  
*Next: [Chapter 14 — Advanced Patterns](ch-14-advanced-patterns.md)*
