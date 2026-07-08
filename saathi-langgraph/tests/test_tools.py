"""Filesystem, search, and shell tools."""

from pathlib import Path

from saathi.tools.filesystem import list_directory, patch_file, read_file, write_file
from saathi.tools.search import search_across_files, search_in_file
from saathi.tools.shell import run_bash


def test_write_then_read(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    result = write_file.invoke({"path": str(f), "content": "hello world"})
    assert "Written" in result
    assert read_file.invoke({"path": str(f)}) == "hello world"


def test_read_missing(tmp_path: Path) -> None:
    assert "not found" in read_file.invoke({"path": str(tmp_path / "nope.txt")})


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = list_directory.invoke({"path": str(tmp_path)})
    assert "a.txt" in out
    assert "sub" in out


def test_patch_missing_file(tmp_path: Path) -> None:
    out = patch_file.invoke({"path": str(tmp_path / "no.txt"), "diff": "irrelevant"})
    assert "not found" in out


def test_search_in_file(tmp_path: Path) -> None:
    f = tmp_path / "c.py"
    f.write_text("import os\nx = 1\nimport sys\n", encoding="utf-8")
    out = search_in_file.invoke({"path": str(f), "pattern": r"^import"})
    assert "import os" in out
    assert "import sys" in out
    assert "1:" in out  # line numbers present


def test_search_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("# TODO fix this\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("nothing here\n", encoding="utf-8")
    out = search_across_files.invoke({"directory": str(tmp_path), "pattern": "TODO"})
    assert "a.py" in out
    assert "TODO" in out


def test_run_bash_echo() -> None:
    out = run_bash.invoke({"command": "echo saathi_test_marker"})
    assert "saathi_test_marker" in out


def test_run_bash_denylist() -> None:
    out = run_bash.invoke({"command": "rm -rf /"})
    assert "Blocked" in out
