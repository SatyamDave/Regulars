from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from regulars.app import Settings, create_app
from regulars.bot.transport import FakeTransport
from regulars.db.engine import sqlite_memory_engine
from regulars.observability.metrics import BOT_TOOL_CALLS
from regulars.payments.fake_stripe import FakeStripeClient
from regulars.payments.stripe_deposits import StripeDeposits
from regulars.wiring import build_system


@pytest.fixture
def app() -> FastAPI:
    settings = Settings.from_env({"STRIPE_API_KEY": "sk", "STRIPE_WEBHOOK_SECRET": "wh"})
    deposits = StripeDeposits("sk", "wh", client=FakeStripeClient())
    return create_app(settings, deposits=deposits)


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_metrics_route_returns_200_and_prometheus_text(http: httpx.AsyncClient) -> None:
    BOT_TOOL_CALLS.inc(tool="send_message", status="ok")
    r = await http.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "# TYPE regulars_bot_tool_calls_total counter" in body
    assert "regulars_bot_tool_calls_total" in body


async def test_metrics_route_does_not_break_existing_routes(http: httpx.AsyncClient) -> None:
    assert (await http.get("/healthz")).status_code == 200


def test_build_system_wires_request_logging_middleware_and_tool_metrics() -> None:
    engine = sqlite_memory_engine()
    transport = FakeTransport()
    system = build_system(Settings(), engine=engine, transport=transport)
    # RequestLoggingMiddleware was added: user_middleware carries it.
    names = [m.cls.__name__ for m in system.app.user_middleware]
    assert "RequestLoggingMiddleware" in names
    # BotTools got the metrics on_call hook wired through build_runtime.
    assert system.runtime.tools._on_call is not None


def test_build_system_observability_false_opts_out() -> None:
    engine = sqlite_memory_engine()
    transport = FakeTransport()
    system = build_system(Settings(), engine=engine, transport=transport, observability=False)
    names = [m.cls.__name__ for m in system.app.user_middleware]
    assert "RequestLoggingMiddleware" not in names
    assert system.runtime.tools._on_call is None
