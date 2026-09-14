from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import Engine

from regulars.app import Settings, create_app
from regulars.payments.stripe_deposits import StripeDeposits
from tests.payments.fakes import FakeStripeClient
from tests.privacy.conftest import Scenario


def _make_app(engine: Engine | None) -> FastAPI:
    settings = Settings.from_env({"STRIPE_API_KEY": "sk_test", "STRIPE_WEBHOOK_SECRET": "whsec"})
    deposits = StripeDeposits(
        settings.stripe_api_key.get_secret_value(),
        settings.stripe_webhook_secret.get_secret_value(),
        client=FakeStripeClient(),
    )
    return create_app(settings, deposits=deposits, engine=engine)


@pytest.fixture
def app(engine: Engine) -> FastAPI:
    return _make_app(engine)


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_export_route_requires_a_requester_id(
    http: httpx.AsyncClient, scenario: Scenario
) -> None:
    """Security-sanity check: the placeholder auth (regulars.app.require_admin_or_self) at
    minimum refuses a request that names nobody. It does NOT verify that header is genuine — see
    its docstring — so this only proves the gate exists, not that it is secure."""
    r = await http.get(f"/privacy/export/{scenario.a.id}")
    assert r.status_code == 401


async def test_export_route_rejects_a_mismatched_requester(
    http: httpx.AsyncClient, scenario: Scenario
) -> None:
    r = await http.get(
        f"/privacy/export/{scenario.a.id}", headers={"X-Requester-Id": scenario.b.id}
    )
    assert r.status_code == 403


async def test_export_route_allows_self_and_admin(
    http: httpx.AsyncClient, scenario: Scenario
) -> None:
    self_resp = await http.get(
        f"/privacy/export/{scenario.a.id}", headers={"X-Requester-Id": scenario.a.id}
    )
    assert self_resp.status_code == 200
    body = self_resp.json()
    assert body["user_id"] == scenario.a.id
    assert "messages" in body and "persona_versions" in body

    admin_resp = await http.get(
        f"/privacy/export/{scenario.a.id}", headers={"X-Requester-Id": "admin"}
    )
    assert admin_resp.status_code == 200


async def test_export_route_missing_user_is_404(http: httpx.AsyncClient) -> None:
    r = await http.get("/privacy/export/nope", headers={"X-Requester-Id": "admin"})
    assert r.status_code == 404


async def test_delete_route_returns_report_shape(
    http: httpx.AsyncClient, scenario: Scenario
) -> None:
    r = await http.post(
        f"/privacy/delete/{scenario.a.id}",
        json={"mode": "anonymize"},
        headers={"X-Requester-Id": scenario.a.id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "anonymize"
    assert body["user_id"] == scenario.a.id
    assert body["persona_versions_scrubbed"] == 2


async def test_delete_route_defaults_to_anonymize(
    http: httpx.AsyncClient, scenario: Scenario
) -> None:
    r = await http.post(
        f"/privacy/delete/{scenario.a.id}", json={}, headers={"X-Requester-Id": "admin"}
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "anonymize"


async def test_delete_route_missing_user_is_404(http: httpx.AsyncClient) -> None:
    r = await http.post(
        "/privacy/delete/nope", json={"mode": "hard"}, headers={"X-Requester-Id": "admin"}
    )
    assert r.status_code == 404


async def test_privacy_routes_503_when_engine_not_configured() -> None:
    app = _make_app(engine=None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        r = await c.get("/privacy/export/anyone", headers={"X-Requester-Id": "admin"})
        assert r.status_code == 503
        r2 = await c.post("/privacy/delete/anyone", json={}, headers={"X-Requester-Id": "admin"})
        assert r2.status_code == 503
