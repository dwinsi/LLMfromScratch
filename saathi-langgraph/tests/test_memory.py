"""Two-scope persistent memory store."""

from saathi.memory.store import MemoryStore


def test_save_and_get(isolated_memory: MemoryStore) -> None:
    isolated_memory.save("global", "name", "ashwin")
    assert isolated_memory.get("global", "name") == "ashwin"
    assert isolated_memory.get("project", "name") is None


def test_project_overrides_global_in_prompt(isolated_memory: MemoryStore) -> None:
    isolated_memory.save("global", "stack", "python")
    isolated_memory.save("project", "stack", "langgraph")
    formatted = isolated_memory.format_for_prompt()
    assert "langgraph" in formatted
    assert "python" not in formatted  # project value wins for the same key


def test_delete(isolated_memory: MemoryStore) -> None:
    isolated_memory.save("project", "k", "v")
    assert isolated_memory.delete("project", "k") is True
    assert isolated_memory.delete("project", "k") is False


def test_clear(isolated_memory: MemoryStore) -> None:
    isolated_memory.save("global", "a", "1")
    isolated_memory.save("global", "b", "2")
    isolated_memory.clear("global")
    assert isolated_memory.all()["global"] == {}


def test_format_empty(isolated_memory: MemoryStore) -> None:
    assert isolated_memory.format_for_prompt() == ""


def test_corrupt_file_is_tolerated(isolated_memory: MemoryStore) -> None:
    isolated_memory._global_path.parent.mkdir(parents=True, exist_ok=True)
    isolated_memory._global_path.write_text("{ broken", encoding="utf-8")
    # Should not raise; corrupt store reads as empty.
    assert isolated_memory.all()["global"] == {}
