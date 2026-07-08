"""System prompt assembly."""

from saathi.agent.prompts import build_system_prompt


def test_base_prompt_present() -> None:
    p = build_system_prompt([], "", "default")
    assert "Saathi" in p


def test_mode_addendum_injected() -> None:
    assert "MODE: explain" in build_system_prompt([], "", "explain")
    assert "MODE: refactor" in build_system_prompt([], "", "refactor")
    assert "MODE: debug" in build_system_prompt([], "", "debug")


def test_unknown_mode_ignored() -> None:
    p = build_system_prompt([], "", "banana")
    assert "MODE:" not in p


def test_memory_block_injected() -> None:
    p = build_system_prompt([], "  entry: main.py", "default")
    assert "Remembered facts" in p
    assert "entry: main.py" in p


def test_context_paths_injected() -> None:
    p = build_system_prompt(["src/", "tests/"], "", "default")
    assert "Context scope" in p
    assert "src/" in p
    assert "tests/" in p


def test_project_instructions_injected() -> None:
    p = build_system_prompt([], "", "default", project_instructions="USE TABS NOT SPACES")
    assert "Project instructions" in p
    assert "USE TABS NOT SPACES" in p
