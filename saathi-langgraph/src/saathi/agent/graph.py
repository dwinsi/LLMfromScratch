"""LangGraph state graph construction."""

import contextlib
from pathlib import Path

import aiosqlite
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from saathi.agent.nodes import make_agent_node
from saathi.agent.state import AgentState
from saathi.agent.tool_node import make_hooked_tool_node
from saathi.config import settings
from saathi.hooks.runner import HookRunner
from saathi.memory.store import MemoryStore


def make_llm(model_id: str, *, json_format: bool = False) -> ChatOllama:
    """Build an (unbound) ChatOllama from settings — used for the agent, history
    summarization, and code review. Set ``json_format`` to force JSON output
    (Ollama's format mode), used by the reviewers."""
    kwargs: dict = dict(
        model=model_id,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.context_window,
        num_predict=settings.max_tokens,
    )
    if json_format:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


async def build_graph(
    tools: list[BaseTool],
    memory_store: MemoryStore,
    model_id: str | None = None,
    db_path: Path | None = None,
    hook_runner: HookRunner | None = None,
):
    """Build and compile the agent graph with an async SQLite checkpointer."""

    model_id = model_id or settings.ollama_model
    llm = make_llm(model_id).bind_tools(tools)

    agent_node = make_agent_node(llm, memory_store)
    tool_node = make_hooked_tool_node(tools, hook_runner or HookRunner())

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    # tools_condition routes agent -> "tools" when the model emitted tool calls,
    # or agent -> END when it produced a final answer. No extra agent->END edge:
    # that would fan out to END on every step and break the ReAct loop.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    db_path = db_path or Path(".saathi") / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # from_conn_string() is an async context manager; for a long-lived REPL we
    # own the connection directly so the saver stays valid for the whole session.
    conn = await aiosqlite.connect(str(db_path))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)


async def close_graph(graph) -> None:
    """Close the SQLite connection backing a compiled graph's checkpointer."""
    checkpointer = getattr(graph, "checkpointer", None)
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        # best-effort cleanup on shutdown
        with contextlib.suppress(Exception):
            await conn.close()
