"""The /doctor health check."""

from saathi.diagnostics import run_doctor


def test_doctor_runs_without_raising() -> None:
    # Must not raise even when Ollama is unreachable (the common CI case).
    run_doctor()
