"""End-to-end test against a real, running Ollama.

Deselected by default (``addopts = -m 'not live'``); run explicitly with::

    pytest -m live

The reachability check runs *inside* the test (not a `skipif` at import time), so
the default suite never touches the network. When Ollama is unreachable or the
configured model isn't pulled, the test **skips** rather than fails — so it's
safe in CI and on machines without Ollama.
"""

from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from saathi.agent import build_graph, close_graph
from saathi.config import settings
from saathi.hooks.runner import HookRunner
from saathi.memory.store import MemoryStore
from saathi.tools import ALL_TOOLS

pytestmark = pytest.mark.live


def _ollama_ready() -> bool:
    """True if Ollama is reachable and the configured model is available."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return False
    want = settings.ollama_model
    return any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)


async def test_live_single_turn(tmp_path: Path) -> None:
    if not _ollama_ready():
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url} or model missing")

    graph = await build_graph(
        ALL_TOOLS,
        MemoryStore(),
        settings.ollama_model,
        db_path=tmp_path / "cp.db",
        hook_runner=HookRunner(),
    )
    try:
        config = {"configurable": {"thread_id": "e2e"}}
        state = {
            "messages": [HumanMessage(content="Reply with exactly the word: PONG")],
            "context_paths": [],
            "mode": "default",
            "session_id": "e2e",
        }
        result = await graph.ainvoke(state, config)
        answers = [
            m
            for m in result["messages"]
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
        ]
        assert answers, "expected at least one assistant response"
        # The model produced a real, non-empty final answer through the full graph.
        assert answers[-1].content.strip()
    finally:
        await close_graph(graph)
