"""Deposit credit ledger (technical-architecture.md §1 ``deposit_credits``; product-plan.md §6).

The ledger is append-only ``LedgerEntry`` rows mirroring the ``deposit_credits`` table
(``user_id, plan_id, amount_cents, kind, created_at``) plus one ``HoldRecord`` per RSVP that
carries the Stripe intent id (``rsvps.deposit_intent_id``) and its lifecycle state.

Entry kinds (amounts are positive except the reversal noted below):

- ``deposit_hold``          an authorization was created for the user on the plan
- ``forfeit``               a no-show's deposit was captured (credit portion included)
- ``redistributed_credit``  the user's share of a plan's forfeited pool, minus the platform cut
- ``applied``               credit spent toward a hold. A *negative* ``applied`` row reverses an
  application: an attendee who paid part of a deposit with credit gets that credit back, the same
  way a cash payer gets their hold released.

``available_credit = sum(redistributed_credit) - sum(applied)``.

Persistence sits behind ``LedgerStore`` so ``regulars.db`` can provide a SQL implementation; the
in-memory store here is for tests and the fake collector. Stripe capture/release goes through the
optional ``DepositGateway`` (``StripeDeposits`` satisfies it). With ``gateway=None`` the ledger
only records, and the caller performs the Stripe calls using the intent ids in ``Settlement``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

LedgerKind = Literal["deposit_hold", "forfeit", "redistributed_credit", "applied"]
HoldState = Literal["held", "captured", "released"]


@dataclass(frozen=True)
class LedgerEntry:
    """One ``deposit_credits`` row."""

    user_id: str
    kind: LedgerKind
    amount_cents: int
    plan_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class HoldRecord:
    """The deposit side of an RSVP: the Stripe hold (if any) plus credit applied toward it."""

    plan_id: str
    user_id: str
    intent_id: str | None
    amount_cents: int
    credit_cents: int = 0
    state: HoldState = "held"

    @property
    def total_cents(self) -> int:
        return self.amount_cents + self.credit_cents


class LedgerStore(Protocol):
    async def add_entry(self, entry: LedgerEntry) -> None: ...

    async def entries(
        self, *, user_id: str | None = None, plan_id: str | None = None
    ) -> list[LedgerEntry]: ...

    async def put_hold(self, hold: HoldRecord) -> None:
        """Insert or replace the hold keyed by ``(plan_id, user_id)``."""
        ...

    async def holds(self, plan_id: str) -> list[HoldRecord]: ...


class InMemoryLedgerStore:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._holds: dict[tuple[str, str], HoldRecord] = {}

    async def add_entry(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    async def entries(
        self, *, user_id: str | None = None, plan_id: str | None = None
    ) -> list[LedgerEntry]:
        return [
            e
            for e in self._entries
            if (user_id is None or e.user_id == user_id)
            and (plan_id is None or e.plan_id == plan_id)
        ]

    async def put_hold(self, hold: HoldRecord) -> None:
        self._holds[(hold.plan_id, hold.user_id)] = hold

    async def holds(self, plan_id: str) -> list[HoldRecord]:
        return [h for h in self._holds.values() if h.plan_id == plan_id]


class DepositGateway(Protocol):
    """The capture/release side of ``StripeDeposits``."""

    async def capture(self, intent_id: str, amount_cents: int | None = None) -> object: ...

    async def release(self, intent_id: str) -> object: ...


class InsufficientCreditError(ValueError):
    pass


@dataclass(frozen=True)
class Settlement:
    """Outcome of ``Ledger.settle_plan``. ``credits`` maps user_id to whole cents credited."""

    plan_id: str
    released: tuple[HoldRecord, ...]
    forfeited: tuple[HoldRecord, ...]
    credits: dict[str, int]
    platform_cents: int

    @property
    def forfeited_cents(self) -> int:
        return sum(h.total_cents for h in self.forfeited)


class Ledger:
    def __init__(
        self,
        store: LedgerStore,
        gateway: DepositGateway | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._gateway = gateway
        self._clock = clock or (lambda: datetime.now(UTC))

    # --- recording -------------------------------------------------------------------------------

    async def hold(
        self,
        plan_id: str,
        user_id: str,
        intent_id: str | None,
        amount_cents: int,
        credit_cents: int = 0,
    ) -> HoldRecord:
        """Record a deposit for an RSVP: ``amount_cents`` on Stripe plus ``credit_cents`` applied.

        A credit-only deposit has ``intent_id=None`` and ``amount_cents=0``.
        """
        if amount_cents < 0 or credit_cents < 0 or amount_cents + credit_cents <= 0:
            raise ValueError("deposit must be positive")
        if (intent_id is None) != (amount_cents == 0):
            raise ValueError("intent_id is required exactly when amount_cents > 0")
        hold = HoldRecord(plan_id, user_id, intent_id, amount_cents, credit_cents)
        await self._store.put_hold(hold)
        await self._add(user_id, "deposit_hold", hold.total_cents, plan_id)
        return hold

    async def available_credit(self, user_id: str) -> int:
        entries = await self._store.entries(user_id=user_id)
        earned = sum(e.amount_cents for e in entries if e.kind == "redistributed_credit")
        spent = sum(e.amount_cents for e in entries if e.kind == "applied")
        return earned - spent

    async def apply_credit(self, user_id: str, plan_id: str, cents: int) -> int:
        """Spend ``cents`` of credit toward ``plan_id``. Returns the amount applied."""
        if cents <= 0:
            raise ValueError("cents must be positive")
        available = await self.available_credit(user_id)
        if cents > available:
            raise InsufficientCreditError(f"{user_id} has {available} credit, asked {cents}")
        await self._add(user_id, "applied", cents, plan_id)
        return cents

    # --- settlement ------------------------------------------------------------------------------

    async def forfeit(self, plan_id: str, user_id: str) -> HoldRecord:
        """Capture a no-show's hold and write the ``forfeit`` row. Idempotent per hold."""
        hold = await self._hold_for(plan_id, user_id)
        if hold.state == "captured":
            return hold
        if hold.state == "released":
            raise ValueError(f"hold for {user_id} on {plan_id} was already released")
        if self._gateway is not None and hold.intent_id is not None:
            await self._gateway.capture(hold.intent_id, hold.amount_cents)
        captured = replace(hold, state="captured")
        await self._store.put_hold(captured)
        await self._add(user_id, "forfeit", hold.total_cents, plan_id)
        return captured

    async def release(self, plan_id: str, user_id: str) -> HoldRecord:
        """Cancel an attendee's hold and hand back any credit applied to it. Idempotent."""
        hold = await self._hold_for(plan_id, user_id)
        if hold.state == "released":
            return hold
        if hold.state == "captured":
            raise ValueError(f"hold for {user_id} on {plan_id} was already captured")
        if self._gateway is not None and hold.intent_id is not None:
            await self._gateway.release(hold.intent_id)
        released = replace(hold, state="released")
        await self._store.put_hold(released)
        if hold.credit_cents:
            await self._add(user_id, "applied", -hold.credit_cents, plan_id)
        return released

    async def settle_plan(
        self,
        plan_id: str,
        attended: Collection[str],
        paid: Collection[str],
        platform_cut_bps: int = 1000,
    ) -> Settlement:
        """Settle every paid hold on the plan and redistribute forfeits to attendees.

        - ``paid`` and ``attended``: hold released.
        - ``paid`` but not ``attended``: hold captured, ``forfeit`` row written.
        - The captured pool minus ``platform_cut_bps`` is split evenly across attendees who paid,
          as ``redistributed_credit`` rows in whole cents; the division remainder goes to the
          platform. With no paying attendees the whole pool stays with the platform.
        - Holds for users not in ``paid`` (never confirmed) are left untouched.

        Idempotent and resumable: hold states gate the Stripe calls and the forfeit/reversal rows,
        and credits are written only once per plan. Re-running returns the same figures.
        """
        if not 0 <= platform_cut_bps <= 10_000:
            raise ValueError("platform_cut_bps must be within [0, 10000]")
        paid_set, attended_set = set(paid), set(attended)
        holds = {h.user_id: h for h in await self._store.holds(plan_id) if h.user_id in paid_set}

        released = [await self.release(plan_id, u) for u in sorted(holds) if u in attended_set]
        forfeited = [await self.forfeit(plan_id, u) for u in sorted(holds) if u not in attended_set]
        pool = sum(h.total_cents for h in forfeited)

        # Resumable: rows already written for this plan win; only missing recipients get a row.
        credits = {
            e.user_id: e.amount_cents
            for e in await self._store.entries(plan_id=plan_id)
            if e.kind == "redistributed_credit"
        }
        for user_id, cents in _split(pool, platform_cut_bps, [h.user_id for h in released]).items():
            if user_id not in credits:
                credits[user_id] = cents
                await self._add(user_id, "redistributed_credit", cents, plan_id)
        return Settlement(
            plan_id=plan_id,
            released=tuple(released),
            forfeited=tuple(forfeited),
            credits=credits,
            platform_cents=pool - sum(credits.values()),
        )

    # --- internals -------------------------------------------------------------------------------

    async def _hold_for(self, plan_id: str, user_id: str) -> HoldRecord:
        for h in await self._store.holds(plan_id):
            if h.user_id == user_id:
                return h
        raise LookupError(f"no hold for {user_id} on plan {plan_id}")

    async def _add(self, user_id: str, kind: LedgerKind, cents: int, plan_id: str | None) -> None:
        await self._store.add_entry(
            LedgerEntry(
                user_id=user_id,
                kind=kind,
                amount_cents=cents,
                plan_id=plan_id,
                created_at=self._clock(),
            )
        )


def _split(pool: int, platform_cut_bps: int, recipients: list[str]) -> dict[str, int]:
    """Even integer split of ``pool`` after the cut; remainder and cut stay with the platform."""
    if pool <= 0 or not recipients:
        return {}
    distributable = pool - (pool * platform_cut_bps) // 10_000
    share = distributable // len(recipients)
    return {u: share for u in recipients} if share > 0 else {}
