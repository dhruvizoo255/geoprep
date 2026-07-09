"""Logging helpers for reproducible discovery runs."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(
    name: str = "geoprep.discovery",
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Create a logger with console output and optional file logging."""

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _parse_level(level: str) -> int:
    parsed = getattr(logging, level.upper(), None)
    if not isinstance(parsed, int):
        return logging.INFO
    return parsed
