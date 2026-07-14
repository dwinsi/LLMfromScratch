# Chapter 6 — Dependency Injection

> **What you'll learn:** what dependency injection is and why it matters, `Depends()`, the `Annotated` shorthand, class-based dependencies, nested dependencies, and a complete walkthrough of `dependencies.py`.

---

## What is dependency injection?

Dependency injection (DI) is a pattern where a function declares what it *needs* rather than creating those things itself. Something external — the DI system — is responsible for constructing and providing the dependencies.

Without DI, every handler that needs the database would create a connection itself:

```python
# Without DI — tightly coupled, hard to test
@app.post("/chat")
async def chat(req: ChatRequest):
    graph = await build_graph(tools=ALL_TOOLS, memory_store=MemoryStore())  # rebuilt every request!
    result = await graph.ainvoke(...)
```

This is bad: the graph is rebuilt on every request (expensive), the handler is coupled to the construction details, and you can't substitute a mock graph in tests.

With DI, the handler declares what it needs and FastAPI provides it:

```python
# With DI — decoupled, testable, efficient
@app.post("/chat")
async def chat(req: ChatRequest, graph: GraphDep):
    result = await graph.ainvoke(...)
```

The handler doesn't know how `graph` was built, where it lives, or how many other handlers share it. FastAPI takes care of all of that.

---

## `Depends()` — the core mechanism

`Depends()` wraps a callable (a function or class). FastAPI calls it to resolve the dependency each time the route is invoked:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_db():
    return {"connection": "database_connection_here"}

@app.get("/items")
def list_items(db = Depends(get_db)):
    return db
```

When `GET /items` is called, FastAPI calls `get_db()`, gets the result, and passes it to `list_items` as `db`. `get_db` can be as simple or complex as needed — connect to a database, read a config, build an object.

> **Experienced note:** Flask has no built-in DI system — you'd use globals, `g`, or an extension like Flask-Injector. DRF has viewset classes but no function-level DI. FastAPI's `Depends()` is arguably its most powerful feature.

---

## The `Annotated` shorthand

Typing `Depends(get_graph)` inline on every handler gets repetitive. Python's `Annotated` type lets you bundle the type and the dependency together into a reusable alias:

```python
from typing import Annotated
from fastapi import Depends

# Define once
GraphDep = Annotated[object, Depends(get_graph)]
MemoryDep = Annotated[object, Depends(get_memory_store)]

# Use everywhere
@app.post("/chat")
async def chat(req: ChatRequest, graph: GraphDep):
    ...

@app.get("/sessions/{id}/history")
async def history(session_id: str, graph: GraphDep):
    ...
```

`Annotated[object, Depends(get_graph)]` means "this parameter has type `object` and should be resolved by calling `get_graph()`". The `object` type is loose — in production you'd use the real type (`CompiledStateGraph` from langgraph) for better static analysis, but `object` avoids importing LangGraph types into the API layer.

---

## Saathi's `dependencies.py` — full walkthrough

```python
# saathi-langgraph/src/saathi/api/dependencies.py

from typing import Annotated
from fastapi import Depends

_graph = None
_memory_store = None

def get_graph():
    if _graph is None:
        raise RuntimeError("Agent graph not initialised — server still starting up.")
    return _graph

def get_memory_store():
    if _memory_store is None:
        raise RuntimeError("Memory store not initialised — server still starting up.")
    return _memory_store

GraphDep = Annotated[object, Depends(get_graph)]
MemoryDep = Annotated[object, Depends(get_memory_store)]
```

**The module-level variables `_graph` and `_memory_store`** are `None` at import time. They're populated during the application's lifespan startup (Chapter 10). This is a deliberate pattern: the graph must be built asynchronously (it opens a database connection), so it can't be created at import time.

**The guard clauses** (`if _graph is None: raise RuntimeError(...)`) prevent routes from running before startup completes. In practice, FastAPI won't serve requests until the lifespan startup finishes, but the guard makes the error message clear if something goes wrong.

**The `Annotated` aliases** (`GraphDep`, `MemoryDep`) are what route handlers actually import and use. Every route that needs the agent graph just declares `graph: GraphDep` — one token, no repetition.

---

## Lifecycle of a dependency

FastAPI's default behaviour is to call the dependency function **once per request**. This means `get_graph()` is called on every request, but because it just returns a module-level object (no construction cost), this is fine.

For expensive dependencies that should be created once and reused (like a database connection pool), you use a **generator dependency** with `yield`:

```python
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session          # ← session is passed to the handler
        await session.commit() # ← runs after the handler returns
```

This is analogous to a context manager. Code before `yield` is setup; code after `yield` is teardown. FastAPI ensures the teardown runs even if the handler raises an exception.

Saathi doesn't need generator dependencies because its graph object is a long-lived singleton — built once during lifespan, shared forever.

---

## Class-based dependencies

For dependencies with configuration, a class is cleaner than a closure:

```python
class Paginator:
    def __init__(self, max_page_size: int = 100):
        self.max_page_size = max_page_size

    def __call__(self, skip: int = 0, limit: int = 10) -> dict:
        limit = min(limit, self.max_page_size)
        return {"skip": skip, "limit": limit}

pagination = Paginator(max_page_size=50)

@app.get("/items")
def list_items(page: dict = Depends(pagination)):
    return page
```

The `__call__` method makes the instance callable, so `Depends(pagination)` works. Parameters to `__call__` are resolved the same way — `skip` and `limit` become query parameters automatically.

---

## Nested dependencies

Dependencies can depend on other dependencies. FastAPI resolves the tree:

```python
def get_settings():
    return {"api_key": "secret"}

def get_client(settings = Depends(get_settings)):
    return SomeClient(api_key=settings["api_key"])

@app.get("/data")
def get_data(client = Depends(get_client)):
    return client.fetch()
```

FastAPI walks the dependency graph: to resolve `client`, it first resolves `settings`. If multiple handlers depend on `get_settings`, it is called once per request (not once per dependent).

---

## Dependencies for cross-cutting concerns

DI is not just for injecting services — it's also ideal for cross-cutting concerns like authentication:

```python
from fastapi import Depends, HTTPException, Header

async def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != "my-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/chat")
async def chat(req: ChatRequest, _: str = Depends(require_api_key)):
    ...
```

Notice `_: str = Depends(require_api_key)` — the parameter is named `_` because we don't use the return value. The dependency only runs for its side effect (the auth check). If it raises an `HTTPException`, FastAPI returns the error response and never calls the handler.

You can also attach dependencies at the router level so they apply to all routes in the router:

```python
router = APIRouter(dependencies=[Depends(require_api_key)])
```

---

## Summary

- Dependency injection lets handlers declare what they need without constructing it themselves.
- `Depends(callable)` wraps any callable as a dependency. FastAPI calls it before the handler runs.
- `Annotated[Type, Depends(fn)]` creates a reusable alias — cleaner than inline `Depends`.
- Module-level globals populated at startup are a clean pattern for long-lived singletons (the Saathi graph).
- Generator dependencies with `yield` provide setup/teardown (database sessions, file handles).
- Class-based dependencies are useful when the dependency needs configuration.
- `Depends` works at the route level, router level, and can be nested.

---

*Previous: [Chapter 5 — Request Body and Response Models](ch-05-request-response.md)*  
*Next: [Chapter 7 — Async Python in FastAPI](ch-07-async-python.md)*
