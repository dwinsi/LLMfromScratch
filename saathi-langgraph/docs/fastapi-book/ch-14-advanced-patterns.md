# Chapter 14 — Advanced Patterns

> **What you'll learn:** background tasks, environment-based configuration with `pydantic-settings`, OpenAPI customisation, and deploying with Gunicorn + uvicorn workers.

---

## Background tasks

Sometimes you want to do something after you've already sent the response — logging an analytics event, sending an email, triggering a slow job. `BackgroundTasks` lets you schedule work to run after the response is sent without making the client wait:

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def send_webhook(session_id: str, message: str):
    # This runs after the response is already sent
    import httpx
    httpx.post("https://my-webhook.example.com", json={
        "session_id": session_id,
        "message": message,
    })

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: GraphDep, background_tasks: BackgroundTasks):
    result = await graph.ainvoke(_build_input(req), config=_thread_config(req.session_id))
    ...
    response = ChatResponse(session_id=req.session_id, reply=reply, ...)

    # Schedule after response is sent — doesn't block the client
    background_tasks.add_task(send_webhook, req.session_id, req.message)

    return response
```

`BackgroundTasks` is injected by FastAPI when you declare it as a parameter. `add_task(fn, *args, **kwargs)` queues the function. It runs in the same event loop thread after the response bytes are sent.

**Important limitation:** background tasks run in the same process and share the same event loop. They're suitable for quick fire-and-forget operations. For heavy work (training a model, generating a report), use a proper task queue like Celery or arq.

---

## `pydantic-settings` — configuration done right

Saathi uses `pydantic-settings` for configuration. This is the standard FastAPI pattern and worth understanding in full.

```python
# saathi-langgraph/src/saathi/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAATHI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = "gemma4:12b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    context_window: int = 32768
    max_tokens: int = 4096
    max_parallel_tools: int = 8
    ollama_max_retries: int = 3
    brave_api_key: str | None = None
    debug: bool = False

settings = Settings()
```

`BaseSettings` from `pydantic-settings` works like a regular Pydantic model but reads values from:
1. Environment variables (highest priority)
2. `.env` file
3. Field defaults (lowest priority)

With `env_prefix="SAATHI_"`, the field `ollama_model` is read from the environment variable `SAATHI_OLLAMA_MODEL`. This means you can override any setting without touching code:

```bash
SAATHI_OLLAMA_MODEL=llama3.2:3b SAATHI_TEMPERATURE=0.7 uvicorn saathi.api.main:app
```

The `.env` file approach means developers can have local overrides without modifying system environment:

```env
# .env (git-ignored)
SAATHI_OLLAMA_MODEL=phi3:mini
SAATHI_DEBUG=true
```

**Why this beats a plain config file:**
- Type validation on every setting (can't set `SAATHI_TEMPERATURE=abc`)
- Pydantic's `@property` for derived settings (`history_token_budget`)
- Same validation patterns as request bodies — consistent tooling
- Auto-documentation via `settings.model_json_schema()`

---

## Caching the settings object

Creating `Settings()` reads from files and environment on every call. Cache it:

```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

# In tests, override it:
app.dependency_overrides[get_settings] = lambda: Settings(ollama_model="test-model")
```

Using `Depends(get_settings)` instead of `settings` directly makes settings mockable in tests — a significant advantage.

---

## OpenAPI customisation

### Custom title, description, version

```python
app = FastAPI(
    title="Saathi API",
    description="""
REST API for the Saathi LangGraph coding agent.

## Features
- Full ReAct agent with tool use
- Token-by-token streaming (SSE)
- Persistent conversation sessions
- Runs 100% locally via Ollama
    """,
    version="1.0.0",
    contact={"name": "Your Name", "email": "you@example.com"},
    license_info={"name": "MIT"},
)
```

### Custom tag descriptions

```python
app = FastAPI(
    openapi_tags=[
        {"name": "chat", "description": "Send messages and stream responses"},
        {"name": "sessions", "description": "Manage conversation history"},
        {"name": "health", "description": "Operational endpoints"},
        {"name": "model", "description": "Model configuration"},
    ]
)
```

### Adding examples to the docs

```python
from pydantic import Field

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        examples=["What Python files are in this project?", "Refactor this function"],
    )
```

These appear in the Swagger UI "Try it out" panel, making the API much more approachable for new users.

### Hiding internal endpoints

```python
@app.get("/internal/debug", include_in_schema=False)
def debug_info():
    return {"internal": True}
```

`include_in_schema=False` hides the endpoint from `/docs` and `/openapi.json`. The endpoint still works — it's just not advertised.

### Changing the docs URL

```python
app = FastAPI(docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")
```

Or disable the docs entirely in production:

```python
import os
app = FastAPI(
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url=None,
)
```

---

## Response caching with headers

For read-heavy endpoints like `/model/info` that return data that rarely changes, set cache headers:

```python
from fastapi import Response

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(response: Response) -> ModelInfoResponse:
    response.headers["Cache-Control"] = "public, max-age=60"
    return ModelInfoResponse(...)
```

This tells browsers and CDN edges to cache the response for 60 seconds. For chat endpoints, use `Cache-Control: no-cache` or no header (the SSE middleware already sets it).

---

## Deploying with Gunicorn + uvicorn workers

For production, run multiple uvicorn workers under Gunicorn process management:

```bash
pip install gunicorn

gunicorn saathi.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile -
```

**`--workers 4`** — four separate processes, each with its own event loop. Rule of thumb: `(2 × CPU cores) + 1`. For a 2-core server, 5 workers.

**`--worker-class uvicorn.workers.UvicornWorker`** — Gunicorn manages the process lifecycle; uvicorn handles the ASGI event loop inside each worker.

**`--timeout 120`** — Gunicorn kills workers that don't respond within 120 seconds. Set this higher for AI APIs where a single inference can take 30–60 seconds.

> **Important for Saathi:** each worker has its own copy of the graph and its own SQLite connection. Multiple workers will conflict on `.saathi/api_checkpoints.db`. For multi-worker deployment, switch the checkpointer to PostgreSQL (`langgraph-checkpoint-postgres`) or Redis.

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e saathi-langgraph gunicorn

CMD ["gunicorn", "saathi.api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

```bash
docker build -t saathi-api .
docker run -p 8000:8000 \
    -e SAATHI_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    saathi-api
```

`host.docker.internal` is the Docker host's IP — so the container can reach Ollama running on your laptop.

---

## Summary

- `BackgroundTasks` runs functions after the response is sent — good for lightweight fire-and-forget, not heavy jobs.
- `pydantic-settings` reads configuration from environment variables and `.env` files with full type validation. Use `env_prefix` to namespace your settings.
- Cache settings with `@lru_cache` and inject via `Depends(get_settings)` for testability.
- Customise the OpenAPI docs with `openapi_tags`, `description`, `examples`, and `include_in_schema=False`.
- Production deployment: Gunicorn + `UvicornWorker` for multi-process scaling.
- Multi-worker SQLite conflicts: switch to a server-based checkpointer for production.

---

*Previous: [Chapter 13 — Middleware](ch-13-middleware.md)*  
*Next: [Chapter 15 — What's Next](ch-15-whats-next.md)*
