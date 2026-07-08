"""Git tools — error handling without mutating any real repository.

These tests never create commits. They exercise the read-only tools against a
non-repository directory and the "git missing" branch via monkeypatch.
"""

import shutil
from pathlib import Path

import pytest

import saathi.tools.git as git_mod
from saathi.tools.git import git_log, git_status

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _isolate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Stop git from walking up into any real parent repository.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def test_status_non_repo(tmp_path: Path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    out = git_status.invoke({})
    assert "not a git repository" in out.lower()


def test_log_non_repo(tmp_path: Path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    out = git_log.invoke({"n": 3})
    assert "not a git repository" in out.lower()


def test_git_binary_missing(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(git_mod.subprocess, "run", boom)
    assert "not installed" in git_status.invoke({}).lower()
