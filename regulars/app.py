"""Web entrypoint: Stripe and BlueBubbles webhooks, privacy self-service, and a health check.

Inbound edges:

- ``POST /webhooks/stripe``: verify ``Stripe-Signature`` with ``StripeDeposits.verify_webhook``
  and hand the ``StripeEvent`` to a ``PaymentEventSink``. Bad signature is a 400.
- ``POST /webhooks/bluebubbles``: hand the raw JSON object to an ``InboundSink``. No parsing here;
  ``regulars.features.ingest`` and the bot own the BlueBubbles shape. When
  ``BLUEBUBBLES_WEBHOOK_TOKEN`` is set the request must carry it in ``X-Webhook-Token``.
- ``GET /privacy/export/{user_id}``: everything the system knows about that user, as JSON
  (``regulars.privacy.export``; §10, GDPR Art. 15 / CCPA right-to-know).
- ``POST /privacy/delete/{user_id}``: anonymize (default) or hard-delete that user's data
  (``regulars.privacy.deletion``; §10, GDPR Art. 17 / CCPA right-to-delete).
- ``GET /healthz``.

Webhook bodies carry card details and message text: never log them. Handler exceptions
propagate as 500s so Stripe and BlueBubbles retry.

**The privacy routes are gated by ``require_admin_or_self``, which is a placeholder, not real
authentication** — see its docstring. A production deployment MUST wire real auth (verified
session/JWT, a genuine admin role check) onto these two routes before they are exposed publicly.

Run: ``uvicorn --factory regulars.app:app_from_env`` (reads ``Settings.from_env()``) or build your
own with ``create_app(settings, payment_sink=..., inbound_sink=..., engine=...)``.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, SecretStr
from sqlalchemy import Engine

from regulars.db.repo import NotFound
from regulars.payments.stripe_deposits import StripeDeposits, StripeEvent, WebhookSignatureError
from regulars.privacy.deletion import DeletionMode, delete_user_data
from regulars.privacy.export import build_data_export

PLACEHOLDER_ADMIN_REQUESTER_ID = "admin"
"""Sentinel accepted by the auth stub below in place of a real admin role. See its docstring."""


async def require_admin_or_self(user_id: str, x_requester_id: str = Header(default="")) -> str:
    """**Placeholder auth. This is NOT real authentication.**

    It checks only that the caller supplied *some* identifying header, and that it names either
    the path's ``user_id`` (self-service) or the hardcoded ``PLACEHOLDER_ADMIN_REQUESTER_ID``
    sentinel. There is no signature verification, no session or token lookup, and no genuine
    admin-role check: anyone who knows a user's id, or the literal string ``"admin"``, passes this
    check. It exists only so the privacy routes have *some* gate rather than none while real auth
    is being built.

    **A production deployment MUST replace this with real authentication** (a verified
    session/JWT resolved to a user id, and an actual admin-role lookup) before ``/privacy/*`` is
    exposed on a public listener. Do not ship this stub past a private/internal deployment.
    """
    if not x_requester_id:
        raise HTTPException(status_code=401, detail="X-Requester-Id header is required")
    if x_requester_id != user_id and x_requester_id != PLACEHOLDER_ADMIN_REQUESTER_ID:
        raise HTTPException(status_code=403, detail="not authorized for this user")
    return x_requester_id


class DeleteRequestBody(BaseModel):
    mode: DeletionMode = "anonymize"


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
    engine: Engine | None = None,
) -> FastAPI:
    app = FastAPI(title="Regulars", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.deposits = deposits or StripeDeposits(
        settings.stripe_api_key.get_secret_value(),
        settings.stripe_webhook_secret.get_secret_value(),
    )
    app.state.payment_sink = payment_sink or RecordingPaymentSink()
    app.state.inbound_sink = inbound_sink or RecordingInboundSink()
    # ``engine`` backs the privacy routes below. Left unset (None) in a deployment that has not
    # wired a database yet; those routes then answer 503 instead of touching an engine that isn't
    # there. Passed explicitly by the caller (e.g. ``regulars.wiring``), never conjured here.
    app.state.privacy_engine = engine
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

    @app.get("/privacy/export/{user_id}")
    async def export_user_data(
        user_id: str,
        request: Request,
        _requester: str = Depends(require_admin_or_self),
    ) -> dict[str, Any]:
        eng = request.app.state.privacy_engine
        if eng is None:
            raise HTTPException(status_code=503, detail="privacy export is not configured")
        try:
            export = await build_data_export(eng, user_id)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail="user not found") from exc
        parsed: dict[str, Any] = json.loads(export.to_json())
        return parsed

    @app.post("/privacy/delete/{user_id}")
    async def delete_user_data_route(
        user_id: str,
        body: DeleteRequestBody,
        request: Request,
        _requester: str = Depends(require_admin_or_self),
    ) -> dict[str, Any]:
        eng = request.app.state.privacy_engine
        if eng is None:
            raise HTTPException(status_code=503, detail="privacy deletion is not configured")
        try:
            report = await delete_user_data(eng, user_id, mode=body.mode)
        except NotFound as exc:
            raise HTTPException(status_code=404, detail="user not found") from exc
        parsed: dict[str, Any] = json.loads(report.model_dump_json())
        return parsed

    return app


def app_from_env() -> FastAPI:
    return create_app(Settings.from_env())
