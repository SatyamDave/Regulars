"""Stripe manual-capture deposits, the credit ledger, and the deposit collector.

Money flow (product-plan.md §3 "Lock", §6): each member authorizes a $20 hold when a plan locks.
Attendees get their hold released; no-shows are captured and the captured pool, minus a small
platform cut, is redistributed to attendees as *credit toward the next plan*. There are no cash
payouts (credit avoids money transmission). The web entrypoint that receives Stripe and
BlueBubbles webhooks lives in ``regulars.app``.
"""

from regulars.payments.collector import (
    DepositCollector,
    DepositRequest,
    DroppedEvent,
    FakeDepositCollector,
    PaidEvent,
    StripeDepositCollector,
    handle_payment_event,
)
from regulars.payments.ledger import (
    DepositGateway,
    HoldRecord,
    HoldState,
    InMemoryLedgerStore,
    InsufficientCreditError,
    Ledger,
    LedgerEntry,
    LedgerKind,
    LedgerStore,
    Settlement,
)
from regulars.payments.stripe_deposits import (
    DepositHold,
    StripeDeposits,
    StripeEvent,
    WebhookSignatureError,
)

__all__ = [
    "DepositCollector",
    "DepositGateway",
    "DepositHold",
    "DepositRequest",
    "DroppedEvent",
    "FakeDepositCollector",
    "HoldRecord",
    "HoldState",
    "InMemoryLedgerStore",
    "InsufficientCreditError",
    "Ledger",
    "LedgerEntry",
    "LedgerKind",
    "LedgerStore",
    "PaidEvent",
    "Settlement",
    "StripeDepositCollector",
    "StripeDeposits",
    "StripeEvent",
    "WebhookSignatureError",
    "handle_payment_event",
]
