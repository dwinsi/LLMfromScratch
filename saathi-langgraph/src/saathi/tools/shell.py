"""Shell execution tool."""

import shlex
import subprocess
import sys

from langchain_core.tools import tool

_BLOCKED = frozenset(
    [
        "rm -rf /",
        ":(){ :|:& };:",
        "mkfs",
        "dd if=/dev/zero",
        "chmod -R 777 /",
        "> /dev/sda",
    ]
)


@tool
def run_bash(command: str) -> str:
    """
    Run a shell command and return its output (stdout + stderr).
    Timeout: 60 seconds. Avoid destructive one-liners.
    """
    for blocked in _BLOCKED:
        if blocked in command:
            return f"Blocked: command matches safety denylist: '{blocked}'"

    try:
        if sys.platform == "win32":
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
            )
        else:
            proc = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
            )

        out = proc.stdout.strip()
        err = proc.stderr.strip()
        parts: list[str] = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        if proc.returncode != 0:
            parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except Exception as e:
        return f"Error running command: {e}"
