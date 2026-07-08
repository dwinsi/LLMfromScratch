"""File-change snapshots that power the /diff command.

These guard the mechanism whose glue bug once made /diff always report
"no changes": snapshots must capture the *original* content of every file the
agent touches, keeping the earliest version per file.
"""

from pathlib import Path

from saathi.tools.filesystem import (
    clear_turn_snapshots,
    get_turn_snapshots,
    write_file,
)
from saathi.ui import display
from saathi.ui.commands import handle_diff


# ── tool-level snapshot capture ───────────────────────────────────────────────
def test_new_file_snapshot_is_empty(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "new.txt"
    write_file.invoke({"path": str(f), "content": "hello"})
    assert get_turn_snapshots()[str(f.resolve())] == ""


def test_existing_file_snapshot_is_original(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "e.txt"
    f.write_text("ORIGINAL", encoding="utf-8")
    write_file.invoke({"path": str(f), "content": "CHANGED"})
    assert get_turn_snapshots()[str(f.resolve())] == "ORIGINAL"


def test_repeated_writes_keep_first_original(tmp_path: Path) -> None:
    clear_turn_snapshots()
    f = tmp_path / "r.txt"
    f.write_text("ORIGINAL", encoding="utf-8")
    write_file.invoke({"path": str(f), "content": "v1"})
    write_file.invoke({"path": str(f), "content": "v2"})
    # Must still be the true original — not the intermediate "v1".
    assert get_turn_snapshots()[str(f.resolve())] == "ORIGINAL"


def test_clear_empties_snapshots(tmp_path: Path) -> None:
    clear_turn_snapshots()
    write_file.invoke({"path": str(tmp_path / "x.txt"), "content": "y"})
    assert get_turn_snapshots()
    clear_turn_snapshots()
    assert get_turn_snapshots() == {}


# ── /diff rendering ────────────────────────────────────────────────────────────
def test_diff_reports_modification(tmp_path: Path) -> None:
    f = tmp_path / "m.txt"
    f.write_text("new content\n", encoding="utf-8")
    with display.console.capture() as cap:
        handle_diff({str(f): "old content\n"})
    out = cap.get()
    assert "old content" in out
    assert "new content" in out


def test_diff_no_changes(tmp_path: Path) -> None:
    f = tmp_path / "same.txt"
    f.write_text("identical\n", encoding="utf-8")
    with display.console.capture() as cap:
        handle_diff({str(f): "identical\n"})
    assert "No file changes" in cap.get()


def test_diff_reports_deletion(tmp_path: Path) -> None:
    missing = tmp_path / "gone.txt"  # never created on disk
    with display.console.capture() as cap:
        handle_diff({str(missing): "was here\n"})
    assert "Deleted" in cap.get()
