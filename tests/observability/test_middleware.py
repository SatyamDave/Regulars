from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from regulars.observability.logging_config import JsonFormatter
from regulars.observability.metrics import WEBHOOK_REQUESTS
from regulars.observability.middleware import REQUEST_ID_HEADER, RequestLoggingMiddleware


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "ok"}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("kaboom")

    return app


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_adds_x_request_id_header_when_absent(http: httpx.AsyncClient) -> None:
    r = await http.get("/ping")
    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER]


async def test_preserves_client_supplied_request_id(http: httpx.AsyncClient) -> None:
    r = await http.get("/ping", headers={REQUEST_ID_HEADER: "client-supplied-id"})
    assert r.headers[REQUEST_ID_HEADER] == "client-supplied-id"


async def test_logs_one_structured_line_per_request(
    http: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="regulars.http"):
        await http.get("/ping")
    records = [r for r in caplog.records if r.name == "regulars.http"]
    assert len(records) == 1
    record = records[0]
    assert record.http_method == "GET"  # type: ignore[attr-defined]
    assert record.http_path == "/ping"  # type: ignore[attr-defined]
    assert record.http_status == 200  # type: ignore[attr-defined]
    assert record.request_id  # type: ignore[attr-defined]
    assert isinstance(record.duration_ms, float)  # type: ignore[attr-defined]

    # The JSON formatter renders that same record into valid, parseable JSON.
    payload = json.loads(JsonFormatter().format(record))
    assert payload["http_status"] == 200


async def test_records_webhook_metric_by_route_and_status(http: httpx.AsyncClient) -> None:
    before = dict(WEBHOOK_REQUESTS.samples())
    await http.get("/ping")
    after = dict(WEBHOOK_REQUESTS.samples())
    key = ("/ping", "200")
    assert after.get(key, 0) == before.get(key, 0) + 1


async def test_logs_status_500_on_handler_exception(
    http: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="regulars.http"):
        r = await http.get("/boom")
    assert r.status_code == 500
    records = [rec for rec in caplog.records if rec.name == "regulars.http"]
    assert len(records) == 1
    assert records[0].http_status == 500  # type: ignore[attr-defined]
    assert records[0].levelname == "ERROR"
