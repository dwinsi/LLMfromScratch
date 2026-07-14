# FastAPI — A Practical Guide

> Built around **Saathi API** — a real REST wrapper for a LangGraph AI coding agent.
> Every concept is grounded in code you can run, read, and modify.

This book is for two readers. If you have never built an HTTP API before, start at Chapter 0 and read forward — each chapter builds on the last. If you already know Flask, Django REST Framework, or Express and just want to understand FastAPI properly, you can jump to any chapter; the "Experienced note" callouts will orient you quickly.

---

## Chapters

| # | Title | What you learn |
|---|---|---|
| [00](ch-00-introduction.md) | Introduction | Why FastAPI exists, how it compares to Flask/Django, what you'll build |
| [01](ch-01-first-app.md) | Your First FastAPI App | Install, write 10 lines, run, hit with curl, open /docs |
| [02](ch-02-how-it-works.md) | How FastAPI Works Under the Hood | ASGI, Starlette, Pydantic, the request/response cycle |
| [03](ch-03-path-operations.md) | Path Operations | HTTP methods, path params, query params, status codes |
| [04](ch-04-pydantic-models.md) | Pydantic Models | BaseModel, Field, validators, Literal, nested models — the heart of FastAPI |
| [05](ch-05-request-response.md) | Request Body and Response Models | Reading JSON bodies, response_model, auto-serialisation |
| [06](ch-06-dependency-injection.md) | Dependency Injection | Depends(), Annotated shortcuts, class-based deps, nested deps |
| [07](ch-07-async-python.md) | Async Python in FastAPI | async/await, the event loop, concurrent I/O, common mistakes |
| [08](ch-08-routers.md) | Routers — Organising a Real Project | APIRouter, prefix, tags, include_router, sub-applications |
| [09](ch-09-streaming-sse.md) | Streaming Responses and SSE | StreamingResponse, async generators, SSE format, consuming streams |
| [10](ch-10-lifespan.md) | Lifespan — Startup and Shutdown | asynccontextmanager lifespan, resource init/cleanup |
| [11](ch-11-error-handling.md) | Error Handling | HTTPException, 422 validation errors, custom exception handlers |
| [12](ch-12-testing.md) | Testing FastAPI Apps | TestClient, pytest-asyncio, mocking dependencies |
| [13](ch-13-middleware.md) | Middleware | add_middleware, CORS, logging, when to use middleware vs Depends |
| [14](ch-14-advanced-patterns.md) | Advanced Patterns | BackgroundTasks, pydantic-settings, OpenAPI customisation, deployment |
| [15](ch-15-whats-next.md) | What's Next | Databases, auth, WebSockets, Docker, cloud deployment |

---

## The running example — Saathi API

Every chapter uses **Saathi API** as its real-world anchor. Saathi is a REST wrapper around a LangGraph coding agent that runs fully locally via Ollama. Its source lives at:

```
saathi-langgraph/src/saathi/api/
├── main.py          ← FastAPI app + lifespan          (Chapters 1, 10)
├── dependencies.py  ← Depends() singletons            (Chapter 6)
├── schemas.py       ← Pydantic models                 (Chapters 4, 5)
└── routes/
    ├── health.py    ← GET /health                     (Chapters 3, 7)
    ├── model.py     ← GET /model/info                 (Chapter 3)
    ├── chat.py      ← POST /chat + POST /chat/stream  (Chapters 5, 9)
    └── sessions.py  ← POST /sessions + GET history    (Chapters 6, 11)
```

---

## How to use this book

Read it alongside the code. Open `saathi-langgraph/src/saathi/api/` in your editor, then read each chapter. By the end you will understand every line of the Saathi API and be able to build something similar from scratch.

Callout conventions used throughout:

> **Beginner note:** explains something that may confuse someone new to APIs or async Python.

> **Experienced note:** compares to Flask, Django REST Framework, or digs into deeper nuance.
