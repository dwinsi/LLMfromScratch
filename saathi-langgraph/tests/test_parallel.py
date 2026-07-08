"""Parallel tool execution in the hooked tool node."""

import asyncio
import time

from langchain_core.tools import tool

from saathi.agent.tool_node import make_hooked_tool_node
from saathi.hooks.runner import HookConfig, HookRunner
from tests.helpers import ai_with_tool_calls, tool_call

_DELAY = 0.3


@tool
async def slow_echo(label: str) -> str:
    """Sleep then echo the label back."""
    await asyncio.sleep(_DELAY)
    return f"done:{label}"


def _no_hooks() -> HookRunner:
    return HookRunner(HookConfig())


async def test_calls_run_in_parallel() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"c{i}") for i in range(5)]

    start = time.monotonic()
    out = await node(ai_with_tool_calls(calls))
    elapsed = time.monotonic() - start

    assert len(out["messages"]) == 5
    # 5 serial calls would take ~5*_DELAY; parallel should be well under half that.
    assert elapsed < (5 * _DELAY) * 0.5
    assert sorted(m.content for m in out["messages"]) == [f"done:{i}" for i in range(5)]


async def test_every_tool_call_is_answered() -> None:
    node = make_hooked_tool_node([slow_echo], _no_hooks())
    calls = [tool_call("slow_echo", {"label": str(i)}, f"id{i}") for i in range(4)]
    out = await node(ai_with_tool_calls(calls))
    assert {m.tool_call_id for m in out["messages"]} == {f"id{i}" for i in range(4)}


async def test_semaphore_caps_concurrency(monkeypatch) -> None:
    from saathi.config import settings

    monkeypatch.setattr(settings, "max_parallel_tools", 2)

    state = {"active": 0, "peak": 0}
    lock = asyncio.Lock()

    @tool
    async def tracked(i: int) -> str:
        """Track how many copies run at once."""
        async with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        async with lock:
            state["active"] -= 1
        return str(i)

    node = make_hooked_tool_node([tracked], _no_hooks())
    calls = [tool_call("tracked", {"i": i}, f"c{i}") for i in range(6)]
    await node(ai_with_tool_calls(calls))

    assert state["peak"] <= 2


async def test_empty_tool_calls_returns_nothing() -> None:
    from langchain_core.messages import HumanMessage

    node = make_hooked_tool_node([slow_echo], _no_hooks())
    out = await node({"messages": [HumanMessage(content="hi")]})
    assert out["messages"] == []
