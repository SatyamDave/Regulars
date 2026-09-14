"""Web entrypoint: Stripe and BlueBubbles webhooks plus a health check.

Two inbound edges, both thin:

- ``POST /webhooks/stripe``: verify ``Stripe-Signature`` with ``StripeDeposits.verify_webhook``
  and hand the ``StripeEvent`` to a ``PaymentEventSink``. Bad signature is a 400.
- ``POST /webhooks/bluebubbles``: hand the raw JSON object to an ``InboundSink``. No parsing here;
  ``regulars.features.ingest`` and the bot own the BlueBubbles shape. When
  ``BLUEBUBBLES_WEBHOOK_TOKEN`` is set the request must carry it in ``X-Webhook-Token``.
- ``GET /healthz``.

Webhook bodies carry card details and message text: never log them. Handler exceptions
propagate as 500s so Stripe and BlueBubbles retry.

Run: ``uvicorn --factory regulars.app:app_from_env`` (reads ``Settings.from_env()``) or build your
own with ``create_app(settings, payment_sink=..., inbound_sink=...)``.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, SecretStr

from regulars.payments.stripe_deposits import StripeDeposits, StripeEvent, WebhookSignatureError


class Settings(BaseModel):
    stripe_api_key: SecretStr = SecretStr("")
    stripe_webhook_secret: SecretStr = SecretStr("")
    bluebubbles_webhook_token: SecretStr = SecretStr("")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        return cls(
            stripe_api_key=SecretStr(e.get("STRIPE_API_KEY", "")),
            stripe_webhook_secret=SecretStr(e.get("STRIPE_WEBHOOK_SECRET", "")),
            bluebubbles_webhook_token=SecretStr(e.get("BLUEBUBBLES_WEBHOOK_TOKEN", "")),
        )


class PaymentEventSink(Protocol):
    async def handle(self, event: StripeEvent) -> None: ...


class InboundSink(Protocol):
    async def handle(self, payload: dict[str, Any]) -> None: ...


@dataclass
class RecordingPaymentSink:
    """Default sink: keeps the events it saw. Replace with the bot's handler when wiring."""

    events: list[StripeEvent] = field(default_factory=list)

    async def handle(self, event: StripeEvent) -> None:
        self.events.append(event)


@dataclass
class RecordingInboundSink:
    payloads: list[dict[str, Any]] = field(default_factory=list)

    async def handle(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)


def create_app(
    settings: Settings,
    *,
    deposits: StripeDeposits | None = None,
    payment_sink: PaymentEventSink | None = None,
    inbound_sink: InboundSink | None = None,
) -> FastAPI:
    app = FastAPI(title="Regulars", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.deposits = deposits or StripeDeposits(
        settings.stripe_api_key.get_secret_value(),
        settings.stripe_webhook_secret.get_secret_value(),
    )
    app.state.payment_sink = payment_sink or RecordingPaymentSink()
    app.state.inbound_sink = inbound_sink or RecordingInboundSink()
    bluebubbles_token = settings.bluebubbles_webhook_token.get_secret_value()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhooks/stripe")
    async def stripe_webhook(request: Request) -> Response:
        payload = await request.body()
        sig = request.headers.get("Stripe-Signature")
        try:
            event = request.app.state.deposits.verify_webhook(payload, sig)
        except WebhookSignatureError as exc:
            raise HTTPException(status_code=400, detail="invalid signature") from exc
        await request.app.state.payment_sink.handle(event)
        return Response(status_code=200)

    @app.post("/webhooks/bluebubbles")
    async def bluebubbles_webhook(request: Request) -> Response:
        if bluebubbles_token:
            given = request.headers.get("X-Webhook-Token", "")
            if not secrets.compare_digest(given.encode(), bluebubbles_token.encode()):
                raise HTTPException(status_code=401, detail="bad webhook token")
        try:
            payload = json.loads(await request.body())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        await request.app.state.inbound_sink.handle(payload)
        return Response(status_code=200)

    return app


def app_from_env() -> FastAPI:
    return create_app(Settings.from_env())
