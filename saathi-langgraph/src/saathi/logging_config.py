"""Structured logging via structlog.

Quiet by default (WARNING and above); ``--debug`` drops the threshold to DEBUG.
All log output goes to **stderr** so stdout stays clean for ``--print`` and
piping. Use :func:`get_logger` at module scope — it returns a lazy proxy that
binds on first use, so it always reflects the active configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(debug: bool = False) -> None:
    """Configure structlog. Call once at startup, after resolving --debug."""
    level = logging.DEBUG if debug else logging.WARNING
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "saathi") -> Any:
    """Return a structlog logger. Lazy — binds to the active config on first log."""
    return structlog.get_logger(name)
