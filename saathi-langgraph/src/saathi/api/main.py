"""FastAPI application entry point for the Saathi LangGraph API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from saathi.agent.graph import build_graph, close_graph
from saathi.api import dependencies
from saathi.api.routes import chat, health, model, sessions
from saathi.memory.store import MemoryStore
from saathi.tools import ALL_TOOLS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent graph once on startup; tear it down on shutdown."""
    dependencies._memory_store = MemoryStore()
    dependencies._graph = await build_graph(
        tools=ALL_TOOLS,
        memory_store=dependencies._memory_store,
        db_path=Path(".saathi") / "api_checkpoints.db",
    )
    yield
    await close_graph(dependencies._graph)


app = FastAPI(
    title="Saathi API",
    description=(
        "REST API wrapping the Saathi LangGraph coding agent.\n\n"
        "The agent runs fully locally via **Ollama** — no cloud keys required.\n\n"
        "### Quick start\n"
        "```bash\n"
        'curl -X POST http://localhost:8000/chat \\\n'
        '     -H "Content-Type: application/json" \\\n'
        '     -d \'{"message": "What files are in this project?"}\'\n'
        "```"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(model.router)
app.include_router(chat.router)
app.include_router(sessions.router)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_STATIC / "index.html")
