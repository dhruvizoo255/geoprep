"""Structured logging utilities used by every GeoPrep stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines for machine-readable run logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "stage"):
            payload["stage"] = getattr(record, "stage")
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str, log_file: str | None = "logs/geoprep.jsonl", level: str = "INFO") -> logging.Logger:
    """Return a configured structured logger."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    formatter = JsonFormatter()

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        exists = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved
            for handler in logger.handlers
        )
        if not exists:
            file_handler = logging.FileHandler(resolved, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
