"""SAATHI.md discovery and hierarchy loading."""

from pathlib import Path

from saathi.project_context import find_project_instructions, instructions_source


def test_none_found(tmp_path: Path) -> None:
    assert find_project_instructions(tmp_path) == ""
    assert instructions_source(tmp_path) is None


def test_single_file(tmp_path: Path) -> None:
    (tmp_path / "SAATHI.md").write_text("ROOT RULES", encoding="utf-8")
    out = find_project_instructions(tmp_path)
    assert "ROOT RULES" in out
    assert instructions_source(tmp_path) == tmp_path / "SAATHI.md"


def test_hierarchy_nearest_wins_last(tmp_path: Path) -> None:
    (tmp_path / "SAATHI.md").write_text("OUTER RULES", encoding="utf-8")
    sub = tmp_path / "pkg" / "module"
    sub.mkdir(parents=True)
    (sub / "SAATHI.md").write_text("INNER RULES", encoding="utf-8")

    out = find_project_instructions(sub)
    assert "OUTER RULES" in out
    assert "INNER RULES" in out
    # Nearest file is appended last so it takes precedence when read top-to-bottom.
    assert out.index("OUTER RULES") < out.index("INNER RULES")
    # instructions_source returns the nearest file.
    assert instructions_source(sub) == sub / "SAATHI.md"
