"""MCP client: config normalization and a live stdio echo-server round-trip."""

import json
import sys
from pathlib import Path

import pytest

from saathi.agent.tool_node import _result_to_text
from saathi.mcp_client import load_mcp_config, load_mcp_tools

_ECHO_SERVER = Path("examples/mcp_echo_server.py")


# ── result coercion (built-in strings vs MCP content blocks) ───────────────────
def test_result_to_text_passes_strings_through() -> None:
    assert _result_to_text("plain output") == "plain output"


def test_result_to_text_joins_mcp_content_blocks() -> None:
    blocks = [
        {"type": "text", "text": "echo: hi"},
        {"type": "text", "text": "second line"},
    ]
    assert _result_to_text(blocks) == "echo: hi\nsecond line"


def test_result_to_text_falls_back_to_str() -> None:
    assert _result_to_text({"a": 1}) == "{'a': 1}"


# ── config loading / normalization (offline) ──────────────────────────────────
def test_missing_config_is_empty(tmp_path: Path) -> None:
    assert load_mcp_config(tmp_path / "absent.json") == {}


def test_malformed_config_is_empty(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_mcp_config(p) == {}


def test_mcp_servers_wrapper_and_stdio_inference(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(
        json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["-y", "x"]}}}),
        encoding="utf-8",
    )
    conns = load_mcp_config(p)
    assert conns == {"fs": {"transport": "stdio", "command": "npx", "args": ["-y", "x"]}}


def test_flat_mapping_and_url_inference(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"weather": {"url": "http://localhost:8000/mcp/"}}), encoding="utf-8")
    conns = load_mcp_config(p)
    assert conns["weather"]["transport"] == "streamable_http"
    assert conns["weather"]["url"] == "http://localhost:8000/mcp/"


def test_explicit_transport_preserved(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"s": {"transport": "sse", "url": "http://x/sse"}}), encoding="utf-8")
    assert load_mcp_config(p)["s"]["transport"] == "sse"


def test_entry_without_command_or_url_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"broken": {"foo": "bar"}}), encoding="utf-8")
    assert load_mcp_config(p) == {}


async def test_empty_connections_returns_no_tools() -> None:
    assert await load_mcp_tools({}) == []


async def test_unreachable_server_degrades_gracefully() -> None:
    # A command that doesn't exist must not raise — just yields no tools.
    conns = {
        "nope": {"transport": "stdio", "command": "definitely-not-a-real-binary-xyz", "args": []}
    }
    assert await load_mcp_tools(conns) == []


# ── live round-trip against the bundled echo server (offline: pure Python) ─────
@pytest.mark.skipif(not _ECHO_SERVER.is_file(), reason="echo server example missing")
async def test_echo_server_tool_roundtrip() -> None:
    conns = {"echo": {"transport": "stdio", "command": sys.executable, "args": [str(_ECHO_SERVER)]}}
    tools = await load_mcp_tools(conns)
    names = {t.name for t in tools}
    assert "echo" in names
    echo = next(t for t in tools if t.name == "echo")
    result = await echo.ainvoke({"text": "hi"})
    assert "echo: hi" in str(result)
