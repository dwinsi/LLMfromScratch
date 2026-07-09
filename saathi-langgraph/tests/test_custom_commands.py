"""User-defined slash commands from .saathi/commands/*.md."""

from pathlib import Path

from saathi.custom_commands import load_custom_commands, render_command


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert load_custom_commands(tmp_path / "nope") == {}


def test_loads_md_files_keyed_by_lowercased_stem(tmp_path: Path) -> None:
    (tmp_path / "Review.md").write_text("review template", encoding="utf-8")
    (tmp_path / "explain.md").write_text("explain template", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")  # not .md
    cmds = load_custom_commands(tmp_path)
    assert set(cmds) == {"review", "explain"}
    assert cmds["review"] == "review template"


def test_render_substitutes_args_token() -> None:
    out = render_command("Explain: $ARGS please", ["src/app.py"])
    assert out == "Explain: src/app.py please"


def test_render_appends_args_when_no_token() -> None:
    out = render_command("Do the thing.", ["extra", "context"])
    assert out == "Do the thing.\n\nextra context"


def test_render_without_args_and_no_token_is_unchanged() -> None:
    assert render_command("Just do it.", []) == "Just do it."


def test_render_empty_args_with_token_clears_placeholder() -> None:
    assert render_command("Explain: $ARGS", []) == "Explain: "
