"""Load tools from external MCP (Model Context Protocol) servers.

Servers are declared in ``.saathi/mcp.json`` (Claude-Desktop style)::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
        },
        "weather": { "url": "http://localhost:8000/mcp/" }
      }
    }

Their tools are fetched via ``langchain-mcp-adapters`` and merged into the
agent's toolset. Loading is **best-effort**: a missing/broken config or an
unreachable server logs a warning and yields no tools — it never stops Saathi
from starting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from langchain_core.tools import BaseTool

from saathi.logging_config import get_logger

log = get_logger()

_CONFIG_PATH = Path(".saathi") / "mcp.json"


def _normalize(cfg: dict) -> dict | None:
    """Build a valid langchain-mcp-adapters connection from a user config entry."""
    transport = cfg.get("transport")
    if transport is None:
        if "command" in cfg:
            transport = "stdio"
        elif "url" in cfg:
            transport = "streamable_http"
        else:
            return None

    if transport == "stdio":
        if "command" not in cfg:
            return None
        conn = {"transport": "stdio", "command": cfg["command"], "args": list(cfg.get("args", []))}
        if "env" in cfg:
            conn["env"] = cfg["env"]
        if "cwd" in cfg:
            conn["cwd"] = cfg["cwd"]
        return conn

    # http / sse transports
    if "url" not in cfg:
        return None
    conn = {"transport": transport, "url": cfg["url"]}
    if "headers" in cfg:
        conn["headers"] = cfg["headers"]
    return conn


def load_mcp_config(path: Path | None = None) -> dict[str, dict]:
    """Read ``.saathi/mcp.json`` into normalized ``{name: connection}``.

    Accepts either ``{"mcpServers": {...}}`` or a flat ``{name: config}`` mapping.
    Entries without a ``command`` or ``url`` are skipped. Missing/invalid file
    returns ``{}``.
    """
    path = path or _CONFIG_PATH
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("mcp_config_invalid", error=str(exc))
        return {}
    if not isinstance(raw, dict):
        return {}

    servers = raw.get("mcpServers", raw)
    if not isinstance(servers, dict):
        return {}

    connections: dict[str, dict] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        conn = _normalize(cfg)
        if conn is None:
            log.warning("mcp_server_skipped", server=name, reason="missing command/url")
            continue
        connections[name] = conn
    return connections


async def load_mcp_tools(connections: dict[str, dict]) -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools.

    Never raises: any failure logs a warning and returns ``[]`` so a broken
    server cannot prevent startup.
    """
    if not connections:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Our normalized plain dicts structurally match the adapter's connection
        # TypedDicts; cast to satisfy the type checker.
        client = MultiServerMCPClient(cast("dict[str, Any]", connections))
        tools = await client.get_tools()
        log.info("mcp_tools_loaded", servers=len(connections), tools=len(tools))
        return tools
    except Exception as exc:  # noqa: BLE001 — external servers, never fatal
        log.warning("mcp_load_failed", error=str(exc))
        return []
