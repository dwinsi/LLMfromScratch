# Chapter 1 — Your First FastAPI App

> **What you'll learn:** how to install FastAPI, write a minimal app, run it with uvicorn, call it with curl, and navigate the auto-generated docs.

---

## Installation

Create a fresh project folder and a virtual environment:

```bash
mkdir my-api
cd my-api
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install fastapi "uvicorn[standard]"
```

> **Beginner note:** A virtual environment is an isolated copy of Python for your project. It means the packages you install here don't interfere with other projects. Always activate it before running your app.

---

## The minimal app

Create `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello from FastAPI!"}
```

That's it. Five lines including the import.

---

## Running it

```bash
uvicorn main:app --reload
```

You'll see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

`main:app` means "the object named `app` inside the file `main.py`". `--reload` makes uvicorn restart automatically whenever you save a file — essential during development.

---

## Calling your API

Open a second terminal (keep uvicorn running in the first) and try:

```bash
curl http://localhost:8000/hello
```

Response:

```json
{"message":"Hello from FastAPI!"}
```

You can also open `http://localhost:8000/hello` directly in your browser.

> **Beginner note:** `curl` is a command-line tool for making HTTP requests. It comes pre-installed on Mac and Linux. On Windows it is included in PowerShell and Git Bash. If you prefer a GUI, Postman and Insomnia are popular alternatives.

---

## The interactive docs

Open `http://localhost:8000/docs` in your browser. You'll see the **Swagger UI** — a fully interactive web page listing all your endpoints.

Click on `GET /hello`, then click **Try it out**, then **Execute**. It sends the request and shows you the response — all without leaving the browser.

Now open `http://localhost:8000/redoc` — this is **ReDoc**, a cleaner read-only version of the same documentation.

Both are generated automatically from your code. You wrote zero documentation — FastAPI inferred everything from your function name, decorators, and (as you'll see soon) type hints.

> **Experienced note:** This is one of FastAPI's biggest selling points over Flask. In Flask you'd reach for Flask-RESTX or apispec and maintain a separate YAML/JSON schema. In FastAPI the schema is always in sync with the code because it *is* derived from the code.

---

## Line-by-line explanation

```python
from fastapi import FastAPI          # 1
                                     
app = FastAPI()                      # 2

@app.get("/hello")                   # 3
def hello():                         # 4
    return {"message": "Hello!"}     # 5
```

**Line 1** — Import the `FastAPI` class. This is the main object you work with.

**Line 2** — Create an instance of `FastAPI`. You can pass metadata here (title, version, description) that appears in the docs. We'll do that properly in the Saathi app.

**Line 3** — The `@app.get("/hello")` decorator registers this function as the handler for `GET /hello`. When a request arrives at that path with the GET method, FastAPI calls `hello()` and turns its return value into a JSON response.

**Line 4** — The handler function. It can be a regular `def` or an `async def` — Chapter 7 covers when to use each.

**Line 5** — Returning a Python dict. FastAPI automatically serialises it to JSON and sets `Content-Type: application/json`. You never call `json.dumps()` or `jsonify()`.

---

## Adding a parameter

Let's make it more interesting:

```python
@app.get("/hello/{name}")
def hello_name(name: str):
    return {"message": f"Hello, {name}!"}
```

```bash
curl http://localhost:8000/hello/Ashwin
# {"message":"Hello, Ashwin!"}
```

The `{name}` in the path is a **path parameter**. FastAPI extracts it and passes it to the function. The `: str` type hint tells FastAPI to validate that it's a string (it always will be from a URL, but for integers this matters — more in Chapter 3).

---

## Connecting to Saathi

The Saathi API's entry point follows exactly this same pattern, just with more configuration:

```python
# saathi-langgraph/src/saathi/api/main.py

app = FastAPI(
    title="Saathi API",
    description="REST API wrapping the Saathi LangGraph coding agent.",
    version="1.0.0",
    lifespan=lifespan,          # ← startup/shutdown hook (Chapter 10)
)
```

The `title`, `description`, and `version` appear in the Swagger UI header. The `lifespan` parameter is how Saathi initialises its AI agent before the first request arrives — we'll cover that in Chapter 10.

---

## What just happened — the big picture

When you ran `uvicorn main:app --reload`, here is what happened:

1. Python imported `main.py` and executed it — creating the `app` object and registering the `hello` route.
2. Uvicorn started an **ASGI server** listening on port 8000.
3. When your `curl` command arrived, uvicorn passed the raw HTTP request to FastAPI.
4. FastAPI matched the path `/hello` to your decorated function, called it, and turned the dict into a JSON response.
5. Uvicorn sent that response back to your terminal.

The terms ASGI and Starlette will come up in Chapter 2. For now, just know the flow: **request → FastAPI → your function → response**.

---

## Summary

- `FastAPI()` creates the application object.
- `@app.get("/path")` (and `.post`, `.put`, `.delete`) registers a route handler.
- Returning a dict from a handler automatically becomes a JSON response.
- `uvicorn main:app --reload` runs the development server.
- `/docs` gives you a free interactive API explorer.

---

*Previous: [Chapter 0 — Introduction](ch-00-introduction.md)*  
*Next: [Chapter 2 — How FastAPI Works Under the Hood](ch-02-how-it-works.md)*
