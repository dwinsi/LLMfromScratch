"""Structured logging configuration and instrumentation."""

from structlog.testing import capture_logs

from saathi.agent.tool_node import make_hooked_tool_node
from saathi.hooks.runner import HookConfig, HookRunner
from saathi.logging_config import configure_logging, get_logger
from saathi.tools import ALL_TOOLS

from tests.helpers import ai_with_tool_calls, tool_call


def test_configure_logging_runs_at_both_levels() -> None:
    configure_logging(debug=True)
    configure_logging(debug=False)


def test_get_logger_emits_structured_event() -> None:
    with capture_logs() as logs:
        get_logger().warning("something_happened", detail=42)
    assert any(e["event"] == "something_happened" and e.get("detail") == 42 for e in logs)


async def test_tool_node_logs_block() -> None:
    node = make_hooked_tool_node(ALL_TOOLS, HookRunner(HookConfig(block_paths=["*.env"])))
    calls = [tool_call("write_file", {"path": "secret.env", "content": "x"}, "a")]
    with capture_logs() as logs:
        await node(ai_with_tool_calls(calls))
    assert any(e["event"] == "tool_blocked" and e.get("tool") == "write_file" for e in logs)


async def test_tool_node_logs_unknown_tool() -> None:
    node = make_hooked_tool_node(ALL_TOOLS, HookRunner(HookConfig()))
    with capture_logs() as logs:
        await node(ai_with_tool_calls([tool_call("does_not_exist", {}, "a")]))
    assert any(e["event"] == "tool_unknown" for e in logs)
