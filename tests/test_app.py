from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from regulars.app import (
    RecordingInboundSink,
    RecordingPaymentSink,
    Settings,
    create_app,
)
from regulars.payments.collector import PaidEvent, handle_payment_event
from regulars.payments.stripe_deposits import StripeDeposits
from tests.payments.fakes import FakeStripeClient, event_json

SECRET = "whsec_test"


@pytest.fixture
def app() -> FastAPI:
    settings = Settings.from_env(
        {
            "STRIPE_API_KEY": "sk_test_fake",
            "STRIPE_WEBHOOK_SECRET": SECRET,
            "BLUEBUBBLES_WEBHOOK_TOKEN": "bb-token",
        }
    )
    deposits = StripeDeposits(
        settings.stripe_api_key.get_secret_value(),
        settings.stripe_webhook_secret.get_secret_value(),
        client=FakeStripeClient(),
    )
    return create_app(settings, deposits=deposits)


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_settings_from_env_hides_secrets() -> None:
    s = Settings.from_env({"STRIPE_API_KEY": "sk_live_abc"})
    assert s.stripe_api_key.get_secret_value() == "sk_live_abc"
    assert "sk_live_abc" not in repr(s) and "sk_live_abc" not in str(s)
    assert s.stripe_webhook_secret.get_secret_value() == ""


async def test_healthz(http: httpx.AsyncClient) -> None:
    r = await http.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_stripe_webhook_rejects_bad_signature(http: httpx.AsyncClient, app: FastAPI) -> None:
    payload = event_json("payment_intent.amount_capturable_updated", "pi_1", "p1", "u1")
    r = await http.post(
        "/webhooks/stripe", content=payload, headers={"Stripe-Signature": "sig:nope"}
    )
    assert r.status_code == 400
    missing = await http.post("/webhooks/stripe", content=payload)
    assert missing.status_code == 400
    sink = app.state.payment_sink
    assert isinstance(sink, RecordingPaymentSink) and sink.events == []


async def test_stripe_webhook_dispatches_verified_event(
    http: httpx.AsyncClient, app: FastAPI
) -> None:
    payload = event_json("payment_intent.amount_capturable_updated", "pi_1", "p1", "u1")
    r = await http.post(
        "/webhooks/stripe", content=payload, headers={"Stripe-Signature": f"sig:{SECRET}"}
    )
    assert r.status_code == 200
    sink = app.state.payment_sink
    assert isinstance(sink, RecordingPaymentSink)
    [event] = sink.events
    assert event.type == "payment_intent.amount_capturable_updated"
    assert handle_payment_event(event) == PaidEvent("p1", "u1", "pi_1")


async def test_bluebubbles_webhook_forwards_raw_payload(
    http: httpx.AsyncClient, app: FastAPI
) -> None:
    body = {
        "type": "new-message",
        "data": {"guid": "m-1", "text": "hi", "handle": {"address": "+1"}},
    }
    r = await http.post("/webhooks/bluebubbles", json=body, headers={"X-Webhook-Token": "bb-token"})
    assert r.status_code == 200
    sink = app.state.inbound_sink
    assert isinstance(sink, RecordingInboundSink) and sink.payloads == [body]


async def test_bluebubbles_webhook_requires_token_and_json_object(
    http: httpx.AsyncClient, app: FastAPI
) -> None:
    assert (await http.post("/webhooks/bluebubbles", json={"a": 1})).status_code == 401
    bad = {"X-Webhook-Token": "wrong"}
    assert (await http.post("/webhooks/bluebubbles", json={"a": 1}, headers=bad)).status_code == 401
    ok = {"X-Webhook-Token": "bb-token"}
    assert (
        await http.post("/webhooks/bluebubbles", content=b"nope", headers=ok)
    ).status_code == 400
    assert (await http.post("/webhooks/bluebubbles", json=[1, 2], headers=ok)).status_code == 400
    sink = app.state.inbound_sink
    assert isinstance(sink, RecordingInboundSink) and sink.payloads == []


async def test_bluebubbles_webhook_open_when_no_token_configured() -> None:
    app = create_app(Settings(), deposits=StripeDeposits("sk", "wh", client=FakeStripeClient()))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.post("/webhooks/bluebubbles", json={"a": 1})).status_code == 200
