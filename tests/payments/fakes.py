"""Offline stand-ins for ``stripe.StripeClient``. No network."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import stripe


@dataclass
class FakeIntent:
    id: str
    client_secret: str | None
    status: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakePaymentIntents:
    created: list[dict[str, Any]] = field(default_factory=list)
    captured: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    canceled: list[str] = field(default_factory=list)
    intents: dict[str, FakeIntent] = field(default_factory=dict)

    async def create_async(self, params: Any) -> FakeIntent:
        self.created.append(dict(params))
        pi_id = f"pi_{len(self.created)}"
        intent = FakeIntent(pi_id, f"{pi_id}_secret", "requires_payment_method", dict(params))
        self.intents[pi_id] = intent
        return intent

    async def capture_async(self, intent: str, params: Any = None) -> FakeIntent:
        self.captured.append((intent, dict(params) if params is not None else None))
        pi = self.intents.setdefault(intent, FakeIntent(intent, None, "requires_capture"))
        pi.status = "succeeded"
        return pi

    async def cancel_async(self, intent: str, params: Any = None) -> FakeIntent:
        self.canceled.append(intent)
        pi = self.intents.setdefault(intent, FakeIntent(intent, None, "requires_capture"))
        pi.status = "canceled"
        return pi


@dataclass
class FakeV1:
    payment_intents: FakePaymentIntents = field(default_factory=FakePaymentIntents)


@dataclass
class FakeStripeClient:
    """Signature check: ``sig_header`` must equal ``"sig:" + secret``; payload is plain JSON."""

    v1: FakeV1 = field(default_factory=FakeV1)

    def construct_event(
        self, payload: str | bytes | bytearray, sig_header: str | None, secret: str | None
    ) -> stripe.Event:
        if sig_header != f"sig:{secret}":
            raise stripe.SignatureVerificationError(  # type: ignore[no-untyped-call]
                "bad signature", sig_header or ""
            )
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("payload is not a JSON object")
        return stripe.Event.construct_from(parsed, "sk_test_fake")


def event_json(
    kind: str, intent_id: str, plan_id: str, user_id: str, event_id: str = "evt_1"
) -> bytes:
    return json.dumps(
        {
            "id": event_id,
            "object": "event",
            "type": kind,
            "data": {
                "object": {
                    "id": intent_id,
                    "object": "payment_intent",
                    "metadata": {"plan_id": plan_id, "user_id": user_id},
                }
            },
        }
    ).encode()
