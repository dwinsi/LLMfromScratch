"""A tool node that runs tool calls in parallel, with hooks and path blocking.

Each tool call in the last AI message becomes an independent coroutine:

    check block_paths  →  pre_tool hook  →  execute tool  →  post_tool hook

All coroutines run concurrently under an asyncio semaphore (bounded by
`settings.max_parallel_tools`), so a batch of reads/searches the model emits in
one turn completes in roughly the time of the slowest call rather than the sum.
Every tool_call_id is always answered with exactly one ToolMessage — including
blocked calls and calls that raise — so the following model step never errors.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from saathi.agent.state import AgentState
from saathi.config import settings
from saathi.hooks.runner import HookRunner
from saathi.logging_config import get_logger

log = get_logger()


def _result_to_text(result: object) -> str:
    """Coerce a tool result to text. Built-in tools return strings; MCP tools
    return a list of content blocks like ``[{"type": "text", "text": "..."}]``."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


def make_hooked_tool_node(tools: list[BaseTool], hook_runner: HookRunner):
    tools_by_name = {t.name: t for t in tools}
    semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tools))

    async def _run_one(call: dict) -> ToolMessage:
        name = call["name"]
        args = call.get("args", {})
        call_id = call["id"]

        # 1. sensitive-path guard, then pre_tool hook (either may block)
        reason = hook_runner.check_block(name, args)
        if reason is None:
            reason = await hook_runner.run_pre_tool(name, args)
        if reason is not None:
            log.warning("tool_blocked", tool=name, reason=reason)
            return ToolMessage(
                content=f"BLOCKED: {reason}. The tool was not executed.",
                tool_call_id=call_id,
                name=name,
            )

        # 2. execute the tool
        tool = tools_by_name.get(name)
        if tool is None:
            log.warning("tool_unknown", tool=name)
            return ToolMessage(
                content=f"Error: unknown tool '{name}'.",
                tool_call_id=call_id,
                name=name,
            )
        log.debug("tool_start", tool=name, args=args)
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:  # mirror ToolNode's default error handling
            log.error("tool_error", tool=name, error=str(exc))
            result = f"Error executing {name}: {exc}"
        else:
            log.debug("tool_ok", tool=name)

        # 3. post_tool hook (best-effort side effects)
        await hook_runner.run("post_tool", name, args)

        return ToolMessage(content=_result_to_text(result), tool_call_id=call_id, name=name)

    async def _guarded(call: dict) -> ToolMessage:
        async with semaphore:
            return await _run_one(call)

    async def hooked_tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = list(getattr(last, "tool_calls", []) or [])
        if not tool_calls:
            return {"messages": []}

        messages = await asyncio.gather(*(_guarded(call) for call in tool_calls))
        return {"messages": list(messages)}

    return hooked_tool_node
