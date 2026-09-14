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
- ``GET /metrics``: Prometheus text exposition of the in-process registry
  (``regulars.observability.metrics``).

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
from importlib import metadata as importlib_metadata
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, SecretStr
from sqlalchemy import Engine

from regulars.app_schemas import ErrorResponse, HealthResponse, WebhookAckResponse
from regulars.db.repo import NotFound
from regulars.observability.metrics import render_prometheus_text
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


def _package_version() -> str:
    """``[project].version`` from ``pyproject.toml``, read via the installed distribution.

    Falls back to reading ``pyproject.toml`` directly (e.g. running from a raw checkout without
    an installed/editable ``regulars`` distribution) so the OpenAPI schema never hardcodes a
    version string that can drift from the one in ``pyproject.toml``.
    """
    try:
        return importlib_metadata.version("regulars")
    except importlib_metadata.PackageNotFoundError:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        version = data["project"]["version"]
        if not isinstance(version, str):
            raise TypeError(
                f"pyproject.toml [project].version is not a string: {version!r}"
            ) from None
        return version


# Documents the existing auth mechanisms in the OpenAPI schema. Both are ``auto_error=False`` so
# they only *describe* the header for Swagger UI / integrators; the actual verification (HMAC
# signature check, constant-time token compare) still happens in the handler body, unchanged.
_stripe_signature_scheme = APIKeyHeader(
    name="Stripe-Signature",
    auto_error=False,
    description=(
        "HMAC signature Stripe computes over the raw request body. Verified in-handler via "
        "StripeDeposits.verify_webhook; an invalid or missing signature yields 400."
    ),
)
_bluebubbles_token_scheme = APIKeyHeader(
    name="X-Webhook-Token",
    auto_error=False,
    description=(
        "Shared secret configured via the BLUEBUBBLES_WEBHOOK_TOKEN environment variable. "
        "Compared with a constant-time equality check when a token is configured; if none is "
        "configured the endpoint accepts requests without this header."
    ),
)


class Settings(BaseModel):
    stripe_api_key: SecretStr = SecretStr("")
    stripe_webhook_secret: SecretStr = SecretStr("")
    bluebubbles_webhook_token: SecretStr = SecretStr("")
    # Deploy target, set via ENVIRONMENT (see .github/workflows/deploy.yml and
    # deploy-staging.yml, docs/environments.md). Not currently read by anything in this
    # module; it exists so callers (log setup, a Sentry SDK init if that lands — see
    # https://github.com/.../sentry integration branch, not yet merged as of this writing)
    # can branch on deploy target without adding another env lookup. Defaulting to
    # "development" keeps every existing caller and test unaffected.
    environment: Literal["development", "staging", "production"] = "development"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        return cls(
            stripe_api_key=SecretStr(e.get("STRIPE_API_KEY", "")),
            stripe_webhook_secret=SecretStr(e.get("STRIPE_WEBHOOK_SECRET", "")),
            bluebubbles_webhook_token=SecretStr(e.get("BLUEBUBBLES_WEBHOOK_TOKEN", "")),
            environment=e.get("ENVIRONMENT", "development"),  # type: ignore[arg-type]
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
    app = FastAPI(
        title="Regulars",
        description=(
            "Regulars simulates the conversation, then checks it against what happened at "
            "dinner. This service is the web entrypoint: it receives Stripe payment webhooks "
            "and BlueBubbles (iMessage) inbound-message webhooks, and exposes a health check."
        ),
        version=_package_version(),
        contact={"name": "Regulars", "url": "https://github.com/SatyamDave/Regulars"},
        openapi_tags=[
            {"name": "webhooks", "description": "Inbound Stripe and BlueBubbles webhooks."},
            {"name": "health", "description": "Liveness check."},
        ],
    )
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

    @app.get(
        "/healthz",
        tags=["health"],
        summary="Health check",
        description=(
            "Returns 200 with a static body whenever the process is up and able to serve requests."
        ),
        status_code=200,
        response_model=HealthResponse,
    )
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/metrics",
        tags=["health"],
        summary="Prometheus metrics",
        description="Text-exposition-format snapshot of the in-process metrics registry.",
    )
    async def metrics() -> Response:
        return Response(content=render_prometheus_text(), media_type="text/plain")

    @app.post(
        "/webhooks/stripe",
        tags=["webhooks"],
        summary="Stripe payment webhook",
        description=(
            "Verifies the `Stripe-Signature` header against the raw request body with "
            "`StripeDeposits.verify_webhook`, then hands the resulting `StripeEvent` to the "
            "configured `PaymentEventSink`. Never logs the request body: it may carry card "
            "details. Handler exceptions other than a bad signature propagate as 500s so Stripe "
            "retries the delivery."
        ),
        status_code=200,
        response_model=WebhookAckResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Missing or invalid Stripe-Signature."}
        },
        dependencies=[Depends(_stripe_signature_scheme)],
    )
    async def stripe_webhook(request: Request) -> Response:
        payload = await request.body()
        sig = request.headers.get("Stripe-Signature")
        try:
            event = request.app.state.deposits.verify_webhook(payload, sig)
        except WebhookSignatureError as exc:
            raise HTTPException(status_code=400, detail="invalid signature") from exc
        await request.app.state.payment_sink.handle(event)
        return Response(status_code=200)

    @app.post(
        "/webhooks/bluebubbles",
        tags=["webhooks"],
        summary="BlueBubbles inbound-message webhook",
        description=(
            "Hands the raw JSON body straight through to the configured `InboundSink`; no "
            "parsing happens here (`regulars.features.ingest` and the bot own the BlueBubbles "
            "shape). When `BLUEBUBBLES_WEBHOOK_TOKEN` is set, the request must carry the same "
            "value in `X-Webhook-Token` (compared with a constant-time equality check); when it "
            "is unset the endpoint accepts requests without that header. Never logs the request "
            "body: it may carry message text. Handler exceptions propagate as 500s so "
            "BlueBubbles retries the delivery."
        ),
        status_code=200,
        response_model=WebhookAckResponse,
        responses={
            400: {
                "model": ErrorResponse,
                "description": "Body is missing, not JSON, or not a JSON object.",
            },
            401: {
                "model": ErrorResponse,
                "description": "X-Webhook-Token missing or does not match.",
            },
        },
        dependencies=[Depends(_bluebubbles_token_scheme)],
    )
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
