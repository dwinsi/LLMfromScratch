# Chapter 5 — Request Body and Response Models

> **What you'll learn:** how FastAPI reads JSON request bodies, how `response_model` controls what gets sent back, automatic serialisation, filtering extra fields, and a complete walkthrough of `POST /chat`.

---

## Reading a request body

When a client sends a `POST` request with a JSON body, FastAPI reads it and validates it against a Pydantic model. You declare this by making one of your handler parameters a Pydantic model type:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

FastAPI knows `item` is a request body (not a path/query parameter) because its type is a `BaseModel` subclass. There is no `@request.json()` call, no `request.get_json()`, no manual parsing.

Try it:

```bash
curl -X POST http://localhost:8000/items \
     -H "Content-Type: application/json" \
     -d '{"name": "Notebook", "price": 9.99}'
# {"name":"Notebook","price":9.99}
```

Send bad data:

```bash
curl -X POST http://localhost:8000/items \
     -H "Content-Type: application/json" \
     -d '{"name": "Notebook", "price": "not-a-number"}'
```

Response (422 Unprocessable Entity):

```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "price"],
      "msg": "Input should be a valid number, unable to parse string as a number",
      "input": "not-a-number"
    }
  ]
}
```

The error is precise: which field failed (`price`), where it came from (`body`), what was wrong, and what was sent. You wrote zero error-handling code.

> **Experienced note:** Flask requires `request.get_json()` and manual validation. DRF uses serializers that do similar work but require more ceremony. FastAPI's approach is the most concise.

---

## `response_model` — controlling the response

`response_model` tells FastAPI the exact shape of the response it should produce:

```python
from pydantic import BaseModel

class ItemIn(BaseModel):
    name: str
    price: float
    internal_cost: float   # sensitive — should NOT be in the response

class ItemOut(BaseModel):
    name: str
    price: float

@app.post("/items", response_model=ItemOut)
def create_item(item: ItemIn) -> ItemOut:
    # Even if we accidentally return internal_cost, it won't appear in the response
    return ItemOut(name=item.name, price=item.price)
```

`response_model` does three things:

1. **Filters** — any fields on the returned object that are not in `response_model` are stripped out before the response is sent. This prevents accidental data leakage.
2. **Validates** — if the returned data doesn't match `response_model`, FastAPI raises a server-side error rather than sending malformed JSON to clients.
3. **Documents** — the response schema in your OpenAPI docs is derived from `response_model`, so clients know exactly what they'll receive.

---

## The return type annotation

You'll see both `response_model=` and a return type annotation in the Saathi routes:

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: GraphDep) -> ChatResponse:
    ...
    return ChatResponse(session_id=..., reply=..., tool_calls_made=...)
```

Both `response_model=ChatResponse` and `-> ChatResponse` are present. The `response_model` parameter is what FastAPI actually uses for serialisation and filtering. The `-> ChatResponse` return annotation is for your type checker (mypy, pyright) — it gives you editor autocomplete and static analysis. They should always match.

> **Beginner note:** The `-> ChatResponse` syntax means "this function returns a ChatResponse object". It's a hint to the programmer and tools, not enforced by Python at runtime. FastAPI reads it at startup to help generate docs, but `response_model=` is the authoritative one for runtime behaviour.

---

## `POST /chat` — complete walkthrough

```python
# routes/chat.py

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: GraphDep) -> ChatResponse:
    """
    Send a message and receive the agent's full reply.
    """
    result = await graph.ainvoke(
        _build_input(req),
        config=_thread_config(req.session_id),
    )

    messages = result.get("messages", [])
    reply = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        tool_calls_made=_count_tool_calls(messages),
    )
```

**Step by step:**

1. `req: ChatRequest` — FastAPI reads the POST body and validates it into a `ChatRequest`. If `message` is missing or `mode` is not one of the four literals, the request fails with 422 before this line runs.

2. `graph: GraphDep` — the compiled LangGraph agent is injected by FastAPI's dependency system. The handler doesn't know how the graph was built — it just uses it. (Chapter 6 covers this.)

3. `await graph.ainvoke(...)` — the agent runs its full ReAct loop: it may call tools (read files, run bash, search the web) before producing a final answer. This can take several seconds for complex tasks.

4. The loop over `reversed(messages)` — LangGraph's state accumulates all messages (human, tool calls, tool results, AI responses). The last AI message with content is the final answer.

5. `return ChatResponse(...)` — FastAPI serialises this via Pydantic and sends it as JSON.

---

## Helper functions

The three helper functions at the top of `chat.py` are worth explaining:

```python
def _build_input(req: ChatRequest) -> dict:
    return {
        "messages": [HumanMessage(content=req.message)],
        "context_paths": req.context_paths,
        "mode": req.mode,
        "session_id": req.session_id,
    }
```

This converts the Pydantic `ChatRequest` into the dict format LangGraph's `AgentState` expects. It wraps the plain string message in a `HumanMessage` object — LangGraph works with LangChain message objects, not raw strings.

```python
def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
```

LangGraph uses thread IDs to look up checkpoint state. This is how conversation history is maintained across multiple calls — the same `session_id` always maps to the same checkpoint thread.

```python
def _count_tool_calls(messages: list) -> int:
    return sum(
        len(m.tool_calls)
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
    )
```

Counts how many tools the agent invoked during this turn. Returned as `tool_calls_made` in the response — useful for debugging and monitoring.

---

## `response_model_exclude_unset`

Sometimes your response model has many optional fields and you don't want to send all the nulls. `response_model_exclude_unset=True` omits any field that wasn't explicitly set on the returned object:

```python
@app.get("/items/{id}", response_model=ItemOut, response_model_exclude_unset=True)
def get_item(id: int) -> ItemOut:
    return ItemOut(name="Notebook")  # price not set → won't appear in response
```

This is useful for partial update (PATCH) responses where only changed fields should be returned.

---

## Returning different status codes from the same endpoint

If you need to return different status codes from within the handler (not just the success code), use `Response` directly:

```python
from fastapi import Response
from fastapi.responses import JSONResponse

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: GraphDep, response: Response):
    if some_condition:
        return JSONResponse(status_code=202, content={"status": "queued"})
    ...
    return ChatResponse(...)
```

Injecting `response: Response` into your handler lets you set headers and status codes imperatively. Use this sparingly — it bypasses `response_model` validation.

---

## Summary

- FastAPI reads the request body when a parameter is typed as a `BaseModel` subclass — no manual parsing.
- Invalid request data returns a 422 with precise field-level error messages.
- `response_model=SomeModel` filters, validates, and documents the response shape.
- The return type annotation `-> SomeModel` is for static type checkers — it should match `response_model`.
- Helper functions that transform data between API shapes and internal types are a clean pattern.

---

*Previous: [Chapter 4 — Pydantic Models](ch-04-pydantic-models.md)*  
*Next: [Chapter 6 — Dependency Injection](ch-06-dependency-injection.md)*
