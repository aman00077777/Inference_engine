"""
Logging utilities for the Fusion framework.

Provides helper functions to configure and retrieve loggers.
"""

import logging
import sys
from typing import Any, Optional

_CONFIGURED = False


def _configure_once() -> None:
    """Configure the root 'fusion' logger exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root = logging.getLogger("fusion")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    _CONFIGURED = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Name of the logger (will be prefixed with 'fusion.').
        level: Logging level (e.g. INFO, DEBUG).

    Returns:
        Configured Logger object.
    """
    _configure_once()
    logger = logging.getLogger(f"fusion.{name}")
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False

    return logger


def setup_logging(log_file: Optional[str] = None,
                  level: str = "INFO") -> None:
    """
    Configure application-wide logging.

    Args:
        log_file: Optional log file path.
        level: Logging level.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured 'key=value' log line. Never pass raw tensors/text here."""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, f"event={event} {parts}".strip())


def kv(**fields: Any) -> str:
    """Format keyword arguments as space-separated key=value pairs.

    None-valued fields are dropped so optional fields (e.g. duration_ms
    before a call completes) do not clutter the log line.

    Example:
        kv(encoder="bert", event="encode_started", batch_size=2)
        -> "encoder=bert event=encode_started batch_size=2"
    """
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


__all__ = ["get_logger", "setup_logging", "log_event", "kv"]
