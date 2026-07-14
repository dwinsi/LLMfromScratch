# Chapter 0 — Introduction

> **What you'll learn:** what FastAPI is, why it was built, how it compares to Flask and Django, and what you'll have built by the end of this book.

---

## The problem FastAPI was built to solve

In 2018, building a Python API meant choosing between two options. You could use **Flask** — minimal, flexible, easy to start with — but it gave you no help with data validation, no auto-generated documentation, and no async support. Or you could use **Django REST Framework** — powerful and feature-rich — but it came with significant ceremony and a steep learning curve.

Both were also fundamentally synchronous. At a time when Python's `asyncio` was maturing and JavaScript's Node.js was proving that async I/O could handle thousands of concurrent connections on cheap hardware, Python web frameworks were still blocking the OS thread for every database call, every HTTP request to another service, every file read.

Sebastián Ramírez (tiangolo) built FastAPI in 2018 to solve all of this at once. It launched in December 2018 and became one of the fastest-growing Python projects ever.

---

## What FastAPI gives you

**1. Speed — twice.**

FastAPI is fast in two senses. First, it is one of the fastest Python web frameworks in benchmarks — comparable to Go and NodeJS — because it runs on **Starlette**, an ASGI framework built for async I/O. Second, it is fast to develop with. A working, validated, documented API endpoint takes a fraction of the code compared to Flask or DRF.

**2. Automatic data validation.**

FastAPI uses **Pydantic** to validate incoming request data. You declare what shape your data should be, and FastAPI checks every incoming request against that shape before your handler runs. Invalid requests are rejected automatically with a clear error message. You never write `if "message" not in body: return 400`.

**3. Automatic API documentation.**

FastAPI generates a full **OpenAPI** specification from your code — no separate YAML files, no manual maintenance. Two interactive UIs are served automatically:
- `/docs` — Swagger UI, where you can test every endpoint in the browser
- `/redoc` — a cleaner, read-only reference

**4. Type safety throughout.**

FastAPI is built on Python's type hint system. You annotate your function parameters and return types, and FastAPI uses those annotations to do validation, serialisation, and documentation. Your editor's autocompletion and type checker work across your entire API.

---

## How it compares

| | Flask | Django REST Framework | FastAPI |
|---|---|---|---|
| Async support | No (Flask 2 has it, rarely used) | No | First-class |
| Data validation | You do it manually | Serializers (verbose) | Pydantic (automatic) |
| API docs | None | DRF Browsable API | Full OpenAPI, Swagger, ReDoc |
| Learning curve | Low | High | Low–Medium |
| Flexibility | Very high | Medium | High |
| Best for | Small apps, simple APIs | Django-integrated REST APIs | Modern async APIs |

> **Experienced note:** If you know Flask, FastAPI will feel familiar in structure (decorators on functions) but very different in philosophy (types everywhere, Pydantic validation, async by default). If you know DRF, you'll find FastAPI much less ceremony — no serializers, no viewsets, no routers with `register()`.

---

## What you'll build

Throughout this book you'll read, understand, and learn to replicate **Saathi API** — a production-quality REST API that wraps an AI coding agent.

Saathi is a LangGraph agent that runs locally via Ollama. It can read files, search code, run shell commands, and answer questions about your codebase. The API you'll study turns this CLI tool into something any HTTP client — a web app, a VS Code extension, another service — can talk to.

Here are the endpoints you'll understand completely by the end:

```
GET  /health                         → is Ollama running?
GET  /model/info                     → what model/config is active?
POST /chat                           → send a message, get the full reply
POST /chat/stream                    → stream the reply token-by-token (SSE)
POST /sessions                       → create a named conversation session
GET  /sessions/{session_id}/history  → get full conversation history
```

Along the way you'll learn every major FastAPI concept: path operations, Pydantic models, dependency injection, async handlers, routers, streaming responses, lifespan events, error handling, testing, and middleware.

---

## Prerequisites

This book assumes:
- You can write Python. Intermediate level is fine — you don't need to know `asyncio` yet.
- You have Python 3.12 installed (3.10+ will work for most chapters).
- You've seen HTTP before — you know what GET vs POST means, and what a JSON payload is.

You do **not** need prior API framework experience. If you've used Flask or Django before, great — the "Experienced note" callouts will help you map concepts across.

---

## Installing FastAPI

```bash
pip install fastapi "uvicorn[standard]"
```

That's two packages:
- `fastapi` — the framework itself
- `uvicorn` — the ASGI server that runs your app (like Gunicorn for WSGI, but async)

The `[standard]` extra for uvicorn pulls in `uvloop` (faster event loop on Linux/Mac) and `websockets` support.

---

## A first look at what FastAPI code looks like

Before diving into concepts, here is the simplest possible FastAPI app:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello, world!"}
```

Run it:

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000/hello` — you get JSON. Open `http://localhost:8000/docs` — you get an interactive UI with your endpoint already documented.

Three lines of logic. Zero boilerplate. That's the promise.

Now let's go understand every part of it.

---

*Next: [Chapter 1 — Your First FastAPI App](ch-01-first-app.md)*
