"""Hooks system — run shell commands around tool calls and turns.

Config lives in `.saathi/hooks.json`:

    {
      "pre_tool":   ["echo about to run $SAATHI_TOOL_NAME"],
      "post_tool":  ["ruff format $SAATHI_TOOL_ARG_PATH"],
      "post_turn":  ["pytest -q"],
      "block_paths": ["*.env", "**/secrets/*", "*.pem"]
    }

- pre_tool / post_tool / post_turn: lists of shell commands run for that event.
  Commands receive context via environment variables (see _build_env).
- pre_tool commands can BLOCK a tool: a non-zero exit code aborts that tool call,
  and the command's stderr/stdout is returned to the model as the reason.
- block_paths: glob patterns; any write_file / patch_file targeting a matching
  path is refused before the tool runs (sensitive-file protection).
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_PATH = Path(".saathi") / "hooks.json"
_PATH_ARG_KEYS = ("path", "file", "filepath", "file_path")


@dataclass
class HookConfig:
    pre_tool: list[str] = field(default_factory=list)
    post_tool: list[str] = field(default_factory=list)
    post_turn: list[str] = field(default_factory=list)
    block_paths: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.pre_tool or self.post_tool or self.post_turn or self.block_paths)


def load_hook_config(path: Path | None = None) -> HookConfig:
    path = path or _CONFIG_PATH
    if not path.is_file():
        return HookConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return HookConfig()
    return HookConfig(
        pre_tool=list(data.get("pre_tool", [])),
        post_tool=list(data.get("post_tool", [])),
        post_turn=list(data.get("post_turn", [])),
        block_paths=list(data.get("block_paths", [])),
    )


@dataclass
class HookResult:
    ok: bool
    output: str


class HookRunner:
    """Executes configured hook commands. Safe no-op when no config is present."""

    def __init__(self, config: HookConfig | None = None) -> None:
        self.config = config or load_hook_config()

    # ── sensitive-file protection ────────────────────────────────────────────
    def check_block(self, tool_name: str, tool_args: dict) -> str | None:
        """Return a reason string if this tool call must be blocked, else None."""
        if tool_name not in ("write_file", "patch_file"):
            return None
        target = _extract_path(tool_args)
        if not target:
            return None
        for pattern in self.config.block_paths:
            if fnmatch.fnmatch(target, pattern) or fnmatch.fnmatch(Path(target).name, pattern):
                return f"path '{target}' matches blocked pattern '{pattern}'"
        return None

    # ── event hooks ───────────────────────────────────────────────────────────
    async def run(
        self, event: str, tool_name: str = "", tool_args: dict | None = None
    ) -> list[HookResult]:
        commands = getattr(self.config, event, [])
        if not commands:
            return []
        env = _build_env(event, tool_name, tool_args or {})
        results: list[HookResult] = []
        for cmd in commands:
            results.append(await _run_command(cmd, env))
        return results

    async def run_pre_tool(self, tool_name: str, tool_args: dict) -> str | None:
        """Fire pre_tool hooks; return a block reason if any command fails."""
        results = await self.run("pre_tool", tool_name, tool_args)
        for r in results:
            if not r.ok:
                return f"pre_tool hook rejected the call: {r.output.strip()}"
        return None


def _extract_path(tool_args: dict) -> str | None:
    for key in _PATH_ARG_KEYS:
        value = tool_args.get(key)
        if isinstance(value, str):
            return value
    return None


def _build_env(event: str, tool_name: str, tool_args: dict) -> dict[str, str]:
    env = dict(os.environ)
    env["SAATHI_EVENT"] = event
    env["SAATHI_TOOL_NAME"] = tool_name
    env["SAATHI_TOOL_ARGS"] = json.dumps(tool_args)
    path = _extract_path(tool_args)
    if path:
        env["SAATHI_TOOL_ARG_PATH"] = path
    return env


async def _run_command(cmd: str, env: dict[str, str]) -> HookResult:
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = (stdout or b"").decode("utf-8", errors="replace")
        return HookResult(ok=proc.returncode == 0, output=output)
    except TimeoutError:
        return HookResult(ok=False, output=f"hook timed out: {cmd}")
    except Exception as e:  # noqa: BLE001
        return HookResult(ok=False, output=f"hook error: {e}")
