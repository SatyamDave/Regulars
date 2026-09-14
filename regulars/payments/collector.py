"""Deposit collection: apply credit first, hold the remainder on Stripe (product-plan.md §6).

``StripeDepositCollector`` is the payments-side implementation of the bot's ``DepositCollector``
(``collect(plan_id, user_id, amount_cents) -> DepositRequest``); ``FakeDepositCollector`` is the
offline stand-in. ``handle_payment_event`` turns a verified ``StripeEvent`` into the two facts the
bot cares about: the member's authorization succeeded (``PaidEvent``) or the hold is gone
(``DroppedEvent``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from regulars.payments.ledger import Ledger
from regulars.payments.stripe_deposits import StripeDeposits, StripeEvent


@dataclass(frozen=True)
class DepositRequest:
    """What to send the member. Both Stripe fields are ``None`` when credit covered the deposit."""

    intent_id: str | None
    pay_url_or_secret: str | None
    amount_after_credit_cents: int


class DepositCollector(Protocol):
    async def collect(self, plan_id: str, user_id: str, amount_cents: int) -> DepositRequest: ...


class StripeDepositCollector:
    def __init__(self, stripe_deposits: StripeDeposits, ledger: Ledger) -> None:
        self._stripe = stripe_deposits
        self._ledger = ledger

    async def collect(
        self,
        plan_id: str,
        user_id: str,
        amount_cents: int,
        customer_id: str | None = None,
    ) -> DepositRequest:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        credit = min(await self._ledger.available_credit(user_id), amount_cents)
        if credit > 0:
            await self._ledger.apply_credit(user_id, plan_id, credit)
        remainder = amount_cents - credit
        if remainder == 0:
            await self._ledger.hold(plan_id, user_id, None, 0, credit_cents=credit)
            return DepositRequest(None, None, 0)
        hold = await self._stripe.create_hold(plan_id, user_id, remainder, customer_id)
        await self._ledger.hold(plan_id, user_id, hold.intent_id, remainder, credit_cents=credit)
        return DepositRequest(hold.intent_id, hold.client_secret, remainder)


@dataclass
class FakeDepositCollector:
    """Records requests and returns deterministic intent ids. No ledger, no Stripe."""

    requests: list[tuple[str, str, int]] = field(default_factory=list)

    async def collect(self, plan_id: str, user_id: str, amount_cents: int) -> DepositRequest:
        self.requests.append((plan_id, user_id, amount_cents))
        intent_id = f"pi_fake_{len(self.requests)}"
        return DepositRequest(intent_id, f"{intent_id}_secret", amount_cents)


@dataclass(frozen=True)
class PaidEvent:
    """Authorization succeeded and the hold is capturable: mark the RSVP paid."""

    plan_id: str
    user_id: str
    intent_id: str


@dataclass(frozen=True)
class DroppedEvent:
    """The hold was canceled: by us at release, by Stripe after 7 days, or by the member."""

    plan_id: str
    user_id: str
    intent_id: str


PAID_EVENT_TYPE = "payment_intent.amount_capturable_updated"
DROPPED_EVENT_TYPE = "payment_intent.canceled"


def handle_payment_event(event: StripeEvent) -> PaidEvent | DroppedEvent | None:
    """Map a verified Stripe event to a bot-facing fact; ``None`` for anything else.

    Events without our ``plan_id``/``user_id`` metadata (intents we did not create) are ignored.
    """
    if event.type not in (PAID_EVENT_TYPE, DROPPED_EVENT_TYPE):
        return None
    meta = event.metadata
    plan_id, user_id, intent_id = meta.get("plan_id"), meta.get("user_id"), event.object.get("id")
    if not plan_id or not user_id or not intent_id:
        return None
    if event.type == PAID_EVENT_TYPE:
        return PaidEvent(plan_id, user_id, str(intent_id))
    return DroppedEvent(plan_id, user_id, str(intent_id))
