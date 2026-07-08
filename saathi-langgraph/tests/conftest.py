"""Shared pytest fixtures for the Saathi test suite."""

from pathlib import Path

import pytest

from saathi.memory.store import MemoryStore


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small readable file with known content, isolated in tmp_path."""
    p = tmp_path / "sample.txt"
    p.write_text("SAMPLE_CONTENT_123", encoding="utf-8")
    return p


@pytest.fixture
def isolated_memory(tmp_path: Path) -> MemoryStore:
    """A MemoryStore whose global/project files live under tmp_path (no real writes)."""
    store = MemoryStore()
    store._global_path = tmp_path / "global" / "memory.json"
    store._project_path = tmp_path / "project" / "memory.json"
    return store
