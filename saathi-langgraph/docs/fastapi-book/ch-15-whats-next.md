# Chapter 15 — What's Next

> **What you'll learn:** the natural next steps after mastering core FastAPI — databases, authentication, WebSockets, and how to keep growing.

---

## What you've covered

By reaching this chapter you understand every line of the Saathi API. You know how FastAPI handles the full request lifecycle, how Pydantic validates data, how dependency injection keeps things decoupled, how async I/O keeps the server responsive, how streaming works, how to test everything, and how to deploy it.

That's a solid foundation. Here's what to tackle next.

---

## Databases — async SQLAlchemy + Alembic

Most real APIs need a database. The standard FastAPI stack is:

- **SQLAlchemy 2.0** — ORM with async support
- **asyncpg** — async PostgreSQL driver
- **Alembic** — database migrations

```bash
pip install sqlalchemy[asyncio] asyncpg alembic
```

A typical async session dependency:

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session          # ← session passed to handler
        await session.commit() # ← runs after handler returns

DbDep = Annotated[AsyncSession, Depends(get_db)]
```

For Saathi, replacing the SQLite checkpointer with PostgreSQL is the key production upgrade — it supports multiple workers and is not file-locked.

---

## Authentication — OAuth2 and JWT

FastAPI has first-class OAuth2 support. The most common pattern for API authentication is **JWT (JSON Web Tokens)**:

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token, return user
    ...

@app.get("/me")
async def read_me(user = Depends(get_current_user)):
    return user
```

FastAPI's official tutorial has an excellent end-to-end OAuth2 + JWT implementation. For Saathi, you'd add auth to protect the chat endpoints if exposing the API externally.

For simpler cases (internal tools, single-user setups), API key authentication via a header is straightforward:

```python
from fastapi import Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

---

## WebSockets

For bidirectional real-time communication (the client can also send messages while the server is streaming), WebSockets are the right choice over SSE:

```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, graph: GraphDep):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            async for chunk, _ in graph.astream(
                {"messages": [HumanMessage(content=message)], ...},
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    await websocket.send_text(chunk.content)
    except Exception:
        await websocket.close()
```

WebSockets are more complex than SSE (you need to handle disconnects, heartbeats, reconnection logic) but enable truly interactive experiences.

---

## Rate limiting

Protect your API from abuse:

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, req: ChatRequest, graph: GraphDep):
    ...
```

For Saathi, rate limiting per IP or per API key prevents a single client from monopolising the local Ollama server.

---

## Caching with Redis

For endpoints that are expensive to compute and return the same result for the same input (e.g., `/model/info`, or a future "summarise this file" endpoint), add response caching:

```bash
pip install redis fastapi-cache2
```

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@contextmanager
async def lifespan(app):
    FastAPICache.init(RedisBackend(redis), prefix="saathi:")
    yield

@app.get("/model/info")
@cache(expire=60)   # cache for 60 seconds
async def model_info():
    return ModelInfoResponse(...)
```

---

## Monitoring and observability

Production APIs need metrics and tracing:

- **Prometheus + Grafana** — metrics (request counts, latencies, error rates)
- **OpenTelemetry** — distributed tracing across services
- **Sentry** — error tracking and alerting

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Adds GET /metrics endpoint for Prometheus to scrape
```

---

## Deploying to the cloud

Once Dockerised (Chapter 14), you can deploy to any container platform:

**Railway** — simplest: connect your GitHub repo, it detects the Dockerfile, deploys automatically.

**Fly.io** — great for apps that need to run close to users geographically.

```bash
fly launch
fly deploy
```

**AWS / GCP / Azure** — ECS, Cloud Run, or AKS for enterprise deployments. The Docker image you built in Chapter 14 works unchanged.

For Saathi specifically: the Ollama LLM server needs a GPU or a fast CPU. Running it in the cloud means provisioning a GPU VM (expensive) or using Ollama's cloud API. For local development and personal tools, running everything on your own machine is usually the right choice.

---

## The FastAPI ecosystem

Libraries that work especially well with FastAPI:

| Library | Purpose |
|---|---|
| `sqlmodel` | ORM that combines SQLAlchemy and Pydantic — great for FastAPI apps |
| `fastapi-users` | Full auth system (JWT, OAuth2, password hashing) |
| `ormar` | Async ORM with Pydantic integration |
| `arq` | Async job queue for background processing |
| `strawberry` | GraphQL for FastAPI |
| `fastapi-pagination` | Cursor/offset pagination helpers |
| `piccolo` | Async ORM with migrations |

---

## Where to go from here

**Read:** The [FastAPI official documentation](https://fastapi.tiangolo.com) is among the best framework docs ever written. Every section is worth reading.

**Build:** Extend the Saathi API — add authentication, a proper database for session storage, a web UI that uses the SSE streaming endpoint. Building on real code you already understand is the fastest way to solidify knowledge.

**Explore:** The [FastAPI GitHub repository](https://github.com/fastapi/fastapi) itself is clean, well-commented Python — worth reading to understand how the decorators and DI system are implemented.

---

## Looking back

You started with a 5-line hello world. By the end of the Saathi API, you're working with:

- Pydantic models with `default_factory`, `Literal`, and optional fields
- Async LangGraph agent invocation with tool use
- Token streaming over SSE with async generators
- Lifespan-managed resources shared via dependency injection
- Structured error responses with custom exception handlers
- A full test suite with mocked dependencies

FastAPI rewards the investment you've made. Its design — types everywhere, validation automatic, async first — pushes you toward code that is readable, testable, and correct. Carry those habits into every project you build.

---

*Previous: [Chapter 14 — Advanced Patterns](ch-14-advanced-patterns.md)*  
*Back to: [Table of Contents](README.md)*
