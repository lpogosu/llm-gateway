"""Structured JSON logging with a request-scoped correlation id.

One line per event, one JSON object per line: that is what a log shipper wants and
what ``jq`` wants. The request id is carried in a ContextVar so that call sites deep
in the pipeline do not have to thread it through their signatures.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from types import TracebackType
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
owner_var: ContextVar[str] = ContextVar("owner", default="-")

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "service": self._service,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "owner": owner_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info is not None:
            payload["exception"] = self._render_exception(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _render_exception(
        self,
        exc_info: tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None],
    ) -> str:
        return self.formatException(exc_info)


def configure_logging(level: str, service: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own colourised handlers; drop them so that every line on
    # stdout is parseable JSON.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
