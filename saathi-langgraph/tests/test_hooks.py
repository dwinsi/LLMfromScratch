"""Hooks: path blocking, pre/post_tool, config loading, env plumbing."""

import os
from pathlib import Path

from saathi.agent.tool_node import make_hooked_tool_node
from saathi.hooks.runner import HookConfig, HookRunner, load_hook_config
from saathi.tools import ALL_TOOLS
from tests.helpers import ai_with_tool_calls, tool_call


# ── block_paths (sensitive-file protection) ───────────────────────────────────
def test_check_block_matches_env() -> None:
    runner = HookRunner(HookConfig(block_paths=["*.env"]))
    assert runner.check_block("write_file", {"path": "config.env"})
    assert runner.check_block("patch_file", {"path": "config.env"})


def test_check_block_only_write_tools() -> None:
    runner = HookRunner(HookConfig(block_paths=["*.env"]))
    assert runner.check_block("read_file", {"path": "config.env"}) is None
    assert runner.check_block("write_file", {"path": "notes.txt"}) is None


# ── pre_tool blocking via exit code ────────────────────────────────────────────
async def test_pre_tool_hook_blocks_on_nonzero_exit(sample_file: Path) -> None:
    runner = HookRunner(HookConfig(pre_tool=["exit 1"]))
    node = make_hooked_tool_node(ALL_TOOLS, runner)
    calls = [tool_call("read_file", {"path": str(sample_file)}, "x")]
    out = await node(ai_with_tool_calls(calls))
    assert out["messages"][0].content.startswith("BLOCKED")
    assert "pre_tool hook rejected" in out["messages"][0].content


async def test_pre_tool_hook_allows_on_zero_exit(sample_file: Path) -> None:
    runner = HookRunner(HookConfig(pre_tool=["exit 0"]))
    node = make_hooked_tool_node(ALL_TOOLS, runner)
    calls = [tool_call("read_file", {"path": str(sample_file)}, "x")]
    out = await node(ai_with_tool_calls(calls))
    assert out["messages"][0].content == "SAMPLE_CONTENT_123"


# ── mixed batch: blocked + ok + unknown all answered ───────────────────────────
async def test_mixed_batch_all_answered(tmp_path: Path, sample_file: Path) -> None:
    runner = HookRunner(HookConfig(block_paths=["*.env"]))
    node = make_hooked_tool_node(ALL_TOOLS, runner)
    calls = [
        tool_call("write_file", {"path": str(tmp_path / "a.env"), "content": "x"}, "a"),
        tool_call("read_file", {"path": str(sample_file)}, "b"),
        tool_call("does_not_exist", {}, "c"),
    ]
    out = await node(ai_with_tool_calls(calls))
    by_id = {m.tool_call_id: m.content for m in out["messages"]}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].startswith("BLOCKED")
    assert by_id["b"] == "SAMPLE_CONTENT_123"
    assert "unknown tool" in by_id["c"]
    # blocked write must not have created the file
    assert not (tmp_path / "a.env").exists()


# ── post_tool shell hook actually runs ─────────────────────────────────────────
async def test_post_tool_hook_executes(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    runner = HookRunner(HookConfig(post_tool=[f'echo hooked > "{marker}"']))
    results = await runner.run("post_tool", "read_file", {"path": "x"})
    assert results and results[0].ok
    assert marker.exists()


async def test_hook_env_vars_available(tmp_path: Path) -> None:
    marker = tmp_path / "env.txt"
    var = "%SAATHI_TOOL_NAME%" if os.name == "nt" else "$SAATHI_TOOL_NAME"
    runner = HookRunner(HookConfig(post_tool=[f'echo {var} > "{marker}"']))
    await runner.run("post_tool", "write_file", {"path": "p"})
    assert "write_file" in marker.read_text(encoding="utf-8")


# ── config loading ─────────────────────────────────────────────────────────────
def test_load_example_config() -> None:
    cfg = load_hook_config(Path("hooks.example.json"))
    assert "*.env" in cfg.block_paths
    assert cfg.post_turn
    assert not cfg.is_empty


def test_load_missing_config_is_empty(tmp_path: Path) -> None:
    cfg = load_hook_config(tmp_path / "absent.json")
    assert cfg.is_empty


def test_load_malformed_config_is_empty(tmp_path: Path) -> None:
    bad = tmp_path / "hooks.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    assert load_hook_config(bad).is_empty
