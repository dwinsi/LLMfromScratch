"""Graph construction, checkpointer, and clean shutdown.

These do not contact Ollama — ChatOllama is instantiated lazily and only called
during an actual turn, so compiling the graph is offline-safe.
"""

from pathlib import Path

from saathi.agent import build_graph, close_graph
from saathi.hooks.runner import HookRunner
from saathi.memory.store import MemoryStore
from saathi.tools import ALL_TOOLS


async def _build(tmp_path: Path):
    return await build_graph(
        ALL_TOOLS,
        MemoryStore(),
        "gemma4:12b",
        db_path=tmp_path / "checkpoints.db",
        hook_runner=HookRunner(),
    )


async def test_graph_compiles_with_expected_nodes(tmp_path: Path) -> None:
    graph = await _build(tmp_path)
    try:
        nodes = list(graph.get_graph().nodes)
        assert "agent" in nodes
        assert "tools" in nodes
    finally:
        await close_graph(graph)


async def test_checkpoint_db_created(tmp_path: Path) -> None:
    db = tmp_path / "checkpoints.db"
    graph = await build_graph(
        ALL_TOOLS, MemoryStore(), "gemma4:12b", db_path=db, hook_runner=HookRunner()
    )
    try:
        assert db.exists()
    finally:
        await close_graph(graph)


async def test_checkpoint_history_queryable(tmp_path: Path) -> None:
    graph = await _build(tmp_path)
    try:
        cfg = {"configurable": {"thread_id": "t1"}}
        history = [s async for s in graph.aget_state_history(cfg)]
        assert history == []  # nothing run yet
    finally:
        await close_graph(graph)


async def test_close_graph_is_idempotent(tmp_path: Path) -> None:
    graph = await _build(tmp_path)
    await close_graph(graph)
    await close_graph(graph)  # second call must not raise


async def test_agent_only_ends_conditionally(tmp_path: Path) -> None:
    """Regression: the agent node must not have an unconditional edge to END.

    tools_condition already routes agent -> tools or agent -> END. An extra
    static agent -> END edge would terminate the ReAct loop on every step.
    """
    graph = await _build(tmp_path)
    try:
        edges = graph.get_graph().edges
        agent_out = [e for e in edges if e.source == "agent"]
        # Both outgoing edges (to tools and to END) must be conditional.
        assert agent_out, "agent has no outgoing edges"
        assert all(e.conditional for e in agent_out), (
            "agent has an unconditional edge; the ReAct loop would break"
        )
    finally:
        await close_graph(graph)
