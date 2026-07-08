"""Environment health checks for the /doctor command."""

import shutil
import sys
from pathlib import Path

import httpx
from rich import box
from rich.table import Table

from saathi.config import settings
from saathi.ui.display import console


def _check_ollama() -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, f"reachable · {len(models)} model(s) installed"
    except Exception as e:
        return False, f"unreachable at {settings.ollama_base_url} ({type(e).__name__})"


def _check_model() -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        want = settings.ollama_model
        if want in models or any(m.split(":")[0] == want.split(":")[0] for m in models):
            return True, f"{want} available"
        return False, f"{want} not pulled — run: ollama pull {want}"
    except Exception:
        return False, "could not query models (Ollama down?)"


def _check_writable(label: str, path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".saathi_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, str(path)
    except OSError as e:
        return False, f"{path} not writable ({e})"


def _check_binary(name: str) -> tuple[bool, str]:
    found = shutil.which(name)
    return (True, found) if found else (False, f"{name} not on PATH")


def run_doctor() -> None:
    """Run all health checks and print a Rich table."""
    checks: list[tuple[str, tuple[bool, str]]] = [
        ("Python", (sys.version_info >= (3, 12), sys.version.split()[0])),
        ("Ollama server", _check_ollama()),
        ("Model", _check_model()),
        ("Global memory dir", _check_writable("global", Path.home() / ".saathi")),
        ("Project memory dir", _check_writable("project", Path(".saathi"))),
        ("git", _check_binary("git")),
        ("patch", _check_binary("patch")),
    ]

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, title="saathi doctor")
    table.add_column("", width=3)
    table.add_column("Check", style="cyan")
    table.add_column("Detail", style="dim")

    all_ok = True
    for label, (ok, detail) in checks:
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(mark, label, detail)
        all_ok = all_ok and ok

    console.print(table)
    if all_ok:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print("[yellow]Some checks failed — see details above.[/yellow]")
