"""Deposit collection seam. ``regulars.payments`` implements it against Stripe; tests use the fake.

The bot never sees a card. It asks the collector for a payment request and forwards the
customer-facing link/secret to the member in a DM. Payment confirmation flows back through
``BotRuntime.payment_received`` when the payments webhook fires.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class DepositRequest:
    intent_id: str
    pay_url_or_secret: str
    amount_after_credit_cents: int


class DepositCollector(Protocol):
    async def collect(self, plan_id: str, user_id: str, amount_cents: int) -> DepositRequest:
        """Create a manual-capture hold for ``amount_cents`` less any credit the user holds."""
        ...


@dataclass
class FakeDepositCollector:
    """Records every request; ``credits`` (user_id -> cents) reduce the amount owed."""

    credits: Mapping[str, int] = field(default_factory=dict)
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    async def collect(self, plan_id: str, user_id: str, amount_cents: int) -> DepositRequest:
        self.calls.append((plan_id, user_id, amount_cents))
        n = len(self.calls)
        owed = max(amount_cents - self.credits.get(user_id, 0), 0)
        return DepositRequest(
            intent_id=f"pi_fake_{n}",
            pay_url_or_secret=f"https://pay.example/{plan_id}/{user_id}/{n}",
            amount_after_credit_cents=owed,
        )
