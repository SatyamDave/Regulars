"""Stripe manual-capture PaymentIntents for plan deposits (product-plan.md §6).

A deposit is an *authorization*, not a charge: ``create_hold`` creates a PaymentIntent with
``capture_method='manual'``; the member confirms it client-side using ``client_secret``, and
Stripe then emits ``payment_intent.amount_capturable_updated``. Settlement either ``capture``s
(no-show, forfeited) or ``release``s (attended, cancels the authorization) each hold.

**Authorization window.** A manual-capture card authorization expires **7 days** after
confirmation; Stripe cancels it automatically and emits ``payment_intent.canceled``. Locking a
plan and settling it must therefore happen inside that window, or the hold has to be
re-authorized with a fresh ``create_hold``. Do not create holds for dinners more than ~6 days
out, and settle promptly after the event.

The Stripe SDK is injected as a *client* satisfying ``StripeClientLike`` (the real
``stripe.StripeClient`` does structurally) so tests run against a fake and never touch the
network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import stripe


class PaymentIntentLike(Protocol):
    """The three PaymentIntent fields we read back."""

    @property
    def id(self) -> str: ...
    @property
    def client_secret(self) -> str | None: ...
    @property
    def status(self) -> str: ...


class PaymentIntentsAPI(Protocol):
    """Subset of ``stripe.StripeClient().v1.payment_intents`` we call."""

    async def create_async(self, params: Any) -> PaymentIntentLike: ...
    async def capture_async(self, intent: str, params: Any = None) -> PaymentIntentLike: ...
    async def cancel_async(self, intent: str, params: Any = None) -> PaymentIntentLike: ...


class V1Like(Protocol):
    @property
    def payment_intents(self) -> PaymentIntentsAPI: ...


class StripeClientLike(Protocol):
    """What ``StripeDeposits`` needs from ``stripe.StripeClient``."""

    @property
    def v1(self) -> V1Like: ...
    def construct_event(
        self, payload: str | bytes | bytearray, sig_header: str | None, secret: str | None
    ) -> stripe.Event: ...


@dataclass(frozen=True)
class DepositHold:
    """A PaymentIntent as the system sees it; ``intent_id`` is ``rsvps.deposit_intent_id``."""

    intent_id: str
    client_secret: str | None
    status: str


@dataclass(frozen=True)
class StripeEvent:
    """A signature-verified webhook event reduced to what the collector needs."""

    id: str
    type: str
    object: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, str]:
        raw = self.object.get("metadata") or {}
        return {str(k): str(v) for k, v in dict(raw).items()}


class WebhookSignatureError(ValueError):
    """The payload was not signed with our webhook secret, or was malformed."""


class StripeDeposits:
    """Manual-capture deposit holds. See the module docstring for the 7-day authorization window."""

    def __init__(
        self, api_key: str, webhook_secret: str, client: StripeClientLike | None = None
    ) -> None:
        self._webhook_secret = webhook_secret
        self._client: StripeClientLike = (
            client if client is not None else stripe.StripeClient(api_key)
        )

    async def create_hold(
        self,
        plan_id: str,
        user_id: str,
        amount_cents: int,
        customer_id: str | None = None,
        currency: str = "usd",
    ) -> DepositHold:
        """Authorize ``amount_cents`` without capturing. Metadata ties the intent to the RSVP."""
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        params: dict[str, Any] = {
            "amount": amount_cents,
            "currency": currency,
            "capture_method": "manual",
            "metadata": {"plan_id": plan_id, "user_id": user_id},
        }
        if customer_id is not None:
            params["customer"] = customer_id
        intent = await self._client.v1.payment_intents.create_async(params)
        return _hold(intent)

    async def capture(self, intent_id: str, amount_cents: int | None = None) -> DepositHold:
        """Capture the authorization (no-show, forfeited). Partial capture when ``amount_cents``."""
        params = {"amount_to_capture": amount_cents} if amount_cents is not None else None
        intent = await self._client.v1.payment_intents.capture_async(intent_id, params)
        return _hold(intent)

    async def release(self, intent_id: str) -> DepositHold:
        """Cancel the authorization (attended; nothing is charged)."""
        intent = await self._client.v1.payment_intents.cancel_async(intent_id)
        return _hold(intent)

    def verify_webhook(self, payload: bytes, sig_header: str | None) -> StripeEvent:
        """Check ``Stripe-Signature`` and parse the event; raises ``WebhookSignatureError``."""
        try:
            event = self._client.construct_event(payload, sig_header, self._webhook_secret)
        except (stripe.SignatureVerificationError, ValueError) as exc:
            raise WebhookSignatureError("invalid Stripe webhook signature or payload") from exc
        raw: dict[str, Any] = event.to_dict()
        data = raw.get("data") or {}
        return StripeEvent(
            id=str(raw["id"]), type=str(raw["type"]), object=dict(data.get("object") or {})
        )


def _hold(intent: PaymentIntentLike) -> DepositHold:
    return DepositHold(
        intent_id=intent.id, client_secret=intent.client_secret, status=intent.status
    )
