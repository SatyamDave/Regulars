"""Offline stand-in for ``stripe.StripeClient`` (technical-architecture.md §9, product-plan §6).

Used by ``regulars.wiring.build_system`` when no real Stripe key is configured, and by tests that
want to exercise the *real* ``POST /webhooks/stripe`` route without a real signature: unlike
``FakeDepositCollector`` (which never talks to a ``StripeDeposits`` at all), this satisfies
``StripeClientLike`` so a real ``StripeDeposits`` can sit behind the route. ``construct_event``
skips signature verification entirely and parses the raw webhook body, which is exactly what
"fake verifier" means here — never use this outside tests or a stripe-less dev environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import stripe

from regulars.payments.stripe_deposits import PaymentIntentLike


class FakeStripeError(ValueError):
    """The fake client was asked to do something a real Stripe call would be needed for."""


@dataclass(frozen=True)
class _FakeIntent:
    id: str
    client_secret: str | None
    status: str


@dataclass(frozen=True)
class _FakeEvent:
    _raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self._raw


@dataclass
class _FakePaymentIntents:
    counter: int = 0

    async def create_async(self, params: Any) -> PaymentIntentLike:
        self.counter += 1
        intent_id = f"pi_fake_{self.counter}"
        return _FakeIntent(intent_id, f"{intent_id}_secret", "requires_capture")

    async def capture_async(self, intent: str, params: Any = None) -> PaymentIntentLike:
        return _FakeIntent(intent, None, "succeeded")

    async def cancel_async(self, intent: str, params: Any = None) -> PaymentIntentLike:
        return _FakeIntent(intent, None, "canceled")


@dataclass
class _FakeV1:
    payment_intents: _FakePaymentIntents = field(default_factory=_FakePaymentIntents)


@dataclass
class FakeStripeClient:
    """``StripeClientLike`` with no network calls. ``v1`` creates deterministic fake intents;
    ``construct_event`` trusts the body verbatim (there is no real signature to check)."""

    v1: _FakeV1 = field(default_factory=_FakeV1)

    def construct_event(
        self, payload: str | bytes | bytearray, sig_header: str | None, secret: str | None
    ) -> stripe.Event:
        """Returns something that only satisfies ``.to_dict()`` (all ``StripeDeposits`` needs),
        cast to ``stripe.Event`` to match ``StripeClientLike`` — never a real ``stripe.Event``."""
        raw = payload.decode() if isinstance(payload, bytes | bytearray) else payload
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise FakeStripeError("payload is not JSON") from exc
        if not isinstance(data, dict):
            raise FakeStripeError("payload must be a JSON object")
        return cast(stripe.Event, _FakeEvent(data))
