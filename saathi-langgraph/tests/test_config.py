"""Settings defaults and derived values."""

from saathi.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ollama_model
    assert s.ollama_base_url.startswith("http")
    assert s.max_parallel_tools == 8
    assert 0.0 <= s.temperature <= 2.0


def test_history_budget_is_75_percent() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.history_token_budget == int(s.context_window * 0.75)


def test_env_prefix_override(monkeypatch) -> None:
    monkeypatch.setenv("SAATHI_MAX_PARALLEL_TOOLS", "3")
    monkeypatch.setenv("SAATHI_OLLAMA_MODEL", "llama3:8b")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.max_parallel_tools == 3
    assert s.ollama_model == "llama3:8b"
