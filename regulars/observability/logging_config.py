"""Stdlib ``logging`` + a JSON formatter, with request-scoped context via ``contextvars``.

No new dependency: :class:`JsonFormatter` is a small ``logging.Formatter`` subclass. Context
(``request_id``, ``user_id``, or anything else a caller binds) rides in a single ``ContextVar``
so it survives across ``await`` points within one request/task but never leaks between
concurrent requests.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar("regulars_log_context", default=None)

# Attributes every ``logging.LogRecord`` carries; anything else on the record is caller-supplied
# via ``extra=`` and belongs in the JSON payload.
_STANDARD_RECORD_KEYS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())


@contextmanager
def bind_context(**kw: Any) -> Iterator[None]:
    """Merge ``kw`` into the current logging context for the duration of the ``with`` block."""
    merged = {**(_CONTEXT.get() or {}), **kw}
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def get_context() -> dict[str, Any]:
    """A copy of the currently bound logging context (e.g. ``request_id``, ``user_id``)."""
    return dict(_CONTEXT.get() or {})


class JsonFormatter(logging.Formatter):
    """Renders one JSON object per record: timestamp, level, logger, message, context, extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(get_context())
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    """Configure the root logger: one stream handler, JSON-formatted unless ``json=False``."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    plain_fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(JsonFormatter() if json else logging.Formatter(plain_fmt))
    root.addHandler(handler)
