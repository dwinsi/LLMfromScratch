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


async def test_fresh_thread_isolates_compacted_history(tmp_path: Path) -> None:
    """Compaction switches thread_id; the new thread must not inherit the old
    thread's accumulated messages (the reason we can't just trim in place)."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    graph = await _build(tmp_path)
    try:
        t1 = {"configurable": {"thread_id": "t1"}}
        await graph.aupdate_state(
            t1,
            {
                "messages": [
                    HumanMessage(content="h1", id="1"),
                    AIMessage(content="a1", id="2"),
                    HumanMessage(content="h2", id="3"),
                    AIMessage(content="a2", id="4"),
                ]
            },
            as_node="agent",
        )
        st1 = await graph.aget_state(t1)
        assert len(st1.values["messages"]) == 4

        # Continue on a fresh thread with a compacted list.
        t2 = {"configurable": {"thread_id": "t2"}}
        await graph.aupdate_state(
            t2,
            {
                "messages": [
                    SystemMessage(content="summary", id="s"),
                    HumanMessage(content="h2", id="3"),
                    AIMessage(content="a2", id="4"),
                ]
            },
            as_node="agent",
        )
        st2 = await graph.aget_state(t2)
        assert [m.content for m in st2.values["messages"]] == ["summary", "h2", "a2"]
        # the old h1/a1 are NOT carried into the new thread
        assert len(st2.values["messages"]) == 3
    finally:
        await close_graph(graph)


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
