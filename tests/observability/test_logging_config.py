from __future__ import annotations

import json
import logging
import sys

from regulars.observability.logging_config import JsonFormatter, bind_context, get_context


def _make_record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="regulars.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_parseable_json_with_expected_fields() -> None:
    formatter = JsonFormatter()
    record = _make_record("hello world", http_status=200)
    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "regulars.test"
    assert "timestamp" in payload
    assert payload["http_status"] == 200


def test_json_formatter_includes_bound_context() -> None:
    formatter = JsonFormatter()
    with bind_context(request_id="req-1", user_id="u-1"):
        payload = json.loads(formatter.format(_make_record()))
    assert payload["request_id"] == "req-1"
    assert payload["user_id"] == "u-1"


def test_bind_context_is_scoped_and_restores_on_exit() -> None:
    assert get_context() == {}
    with bind_context(request_id="outer"):
        assert get_context() == {"request_id": "outer"}
        with bind_context(user_id="inner"):
            assert get_context() == {"request_id": "outer", "user_id": "inner"}
        assert get_context() == {"request_id": "outer"}
    assert get_context() == {}


def test_json_formatter_includes_exception_info() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="regulars.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exc_info"]
