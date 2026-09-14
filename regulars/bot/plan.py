"""The Plan: a structured record, not a message (§9).

The group chat is a *rendering* of this object. Transitions are pure functions
``Plan -> Plan`` (frozen pydantic models; ``model_copy`` under the hood) so that a stored plan
can be replayed and a rendering is always a function of the record alone.

Lifecycle::

    open, poll open        vote()
    open, poll closed      close_poll(now)  -> locked_slot_id chosen (plurality; ties -> earliest)
    open, deposits out     mark_deposit_requested() per member, then mark_paid() as money lands
    open, some dropped     drop_unpaid(now) 24h after deposits went out
    locked                 lock() once every remaining member has paid
    cancelled              cancel()
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from regulars.bot import voice

RSVPState = Literal["invited", "deposit_requested", "paid", "dropped"]
Outcome = Literal["open", "locked", "cancelled"]

DEFAULT_STAKE_CENTS = 2000
DEPOSIT_WINDOW = timedelta(hours=24)
POLL_WINDOW = timedelta(hours=24)


class PlanError(ValueError):
    """A transition was applied to a plan in the wrong state."""


class Slot(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    venue: str
    starts_at: datetime


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    group_id: str
    proposer: str = "bot"
    constraints: dict[str, str] = Field(default_factory=dict)
    candidate_slots: list[Slot]
    votes: dict[str, str] = Field(default_factory=dict, description="user_id -> slot_id")
    commitment: dict[str, RSVPState] = Field(default_factory=dict)
    stake_cents: int = DEFAULT_STAKE_CENTS
    deadline: datetime
    outcome: Outcome = "open"
    locked_slot_id: str | None = None
    poll_closed_at: datetime | None = None
    deposit_requested_at: datetime | None = None
    deposit_intents: dict[str, str] = Field(default_factory=dict, description="user_id -> intent")
    locked_at: datetime | None = None

    # --- queries -------------------------------------------------------------------------------

    @property
    def poll_open(self) -> bool:
        return self.outcome == "open" and self.locked_slot_id is None

    @property
    def locked_slot(self) -> Slot | None:
        if self.locked_slot_id is None:
            return None
        return self.slot(self.locked_slot_id)

    def slot(self, slot_id: str) -> Slot:
        for s in self.candidate_slots:
            if s.id == slot_id:
                return s
        raise PlanError(f"unknown slot {slot_id!r}")

    def members(self) -> list[str]:
        """Everyone still on the plan (not dropped)."""
        return [u for u, s in self.commitment.items() if s != "dropped"]

    def paid(self) -> list[str]:
        return [u for u, s in self.commitment.items() if s == "paid"]

    def unpaid(self) -> list[str]:
        return [u for u, s in self.commitment.items() if s == "deposit_requested"]

    def all_paid(self) -> bool:
        members = self.members()
        return bool(members) and all(self.commitment[u] == "paid" for u in members)

    def deposit_window_closes_at(self) -> datetime | None:
        if self.deposit_requested_at is None:
            return None
        return self.deposit_requested_at + DEPOSIT_WINDOW


# --- transitions -------------------------------------------------------------------------------


def new_plan(
    plan_id: str,
    group_id: str,
    members: Iterable[str],
    slots: Sequence[Slot],
    deadline: datetime,
    *,
    proposer: str = "bot",
    constraints: Mapping[str, str] | None = None,
    stake_cents: int = DEFAULT_STAKE_CENTS,
) -> Plan:
    if not slots:
        raise PlanError("a plan needs at least one candidate slot")
    return Plan(
        id=plan_id,
        group_id=group_id,
        proposer=proposer,
        constraints=dict(constraints or {}),
        candidate_slots=list(slots),
        commitment={u: "invited" for u in members},
        stake_cents=stake_cents,
        deadline=deadline,
    )


def vote(plan: Plan, user_id: str, slot_id: str) -> Plan:
    """Record (or change) one member's vote. Idempotent per user."""
    if not plan.poll_open:
        raise PlanError("poll is closed")
    if user_id not in plan.commitment or plan.commitment[user_id] == "dropped":
        raise PlanError(f"{user_id!r} is not on this plan")
    plan.slot(slot_id)  # validates
    return plan.model_copy(update={"votes": {**plan.votes, user_id: slot_id}})


def close_poll(plan: Plan, now: datetime) -> Plan:
    """Pick the winner by plurality; ties go to the slot that starts earliest.

    With no votes at all the earliest slot wins: a group that did not vote still gets a plan,
    and the deposit step is what filters the flakes.
    """
    if not plan.poll_open:
        raise PlanError("poll already closed")
    tally = Counter(plan.votes.values())
    best = max(tally.values(), default=0)
    contenders = [s for s in plan.candidate_slots if tally.get(s.id, 0) == best]
    winner = min(contenders, key=lambda s: s.starts_at)
    return plan.model_copy(update={"locked_slot_id": winner.id, "poll_closed_at": now})


def tied(plan: Plan) -> bool:
    """True when the closed poll had more than one slot at the top count."""
    if plan.locked_slot_id is None:
        return False
    tally = Counter(plan.votes.values())
    best = max(tally.values(), default=0)
    return sum(1 for s in plan.candidate_slots if tally.get(s.id, 0) == best) > 1


def mark_deposit_requested(plan: Plan, user_id: str, intent_id: str, now: datetime) -> Plan:
    if plan.locked_slot_id is None:
        raise PlanError("close the poll before requesting deposits")
    if plan.commitment.get(user_id) not in ("invited", "deposit_requested"):
        raise PlanError(f"{user_id!r} cannot be asked for a deposit in state {plan.commitment}")
    return plan.model_copy(
        update={
            "commitment": {**plan.commitment, user_id: "deposit_requested"},
            "deposit_intents": {**plan.deposit_intents, user_id: intent_id},
            "deposit_requested_at": plan.deposit_requested_at or now,
        }
    )


def mark_paid(plan: Plan, user_id: str) -> Plan:
    state = plan.commitment.get(user_id)
    if state == "paid":
        return plan
    if state not in ("invited", "deposit_requested"):
        raise PlanError(f"{user_id!r} cannot pay from state {state!r}")
    return plan.model_copy(update={"commitment": {**plan.commitment, user_id: "paid"}})


def drop_unpaid(plan: Plan, now: datetime) -> tuple[Plan, list[str]]:
    """Drop everyone still unpaid once the 24h window has passed. No-op before then."""
    closes = plan.deposit_window_closes_at()
    if closes is None or now < closes or plan.outcome != "open":
        return plan, []
    dropped = plan.unpaid()
    if not dropped:
        return plan, []
    commitment: dict[str, RSVPState] = {
        **plan.commitment,
        **{u: "dropped" for u in dropped},
    }
    return plan.model_copy(update={"commitment": commitment}), dropped


def lock(plan: Plan, now: datetime) -> Plan:
    if plan.outcome != "open":
        raise PlanError(f"cannot lock a plan that is {plan.outcome}")
    if plan.locked_slot_id is None:
        raise PlanError("close the poll before locking")
    if not plan.all_paid():
        raise PlanError("cannot lock while members are unpaid")
    return plan.model_copy(update={"outcome": "locked", "locked_at": now})


def cancel(plan: Plan) -> Plan:
    if plan.outcome == "locked":
        raise PlanError("cannot cancel a locked plan")
    return plan.model_copy(update={"outcome": "cancelled"})


# --- renderers: pure functions of the Plan ----------------------------------------------------


def render_proposal(plan: Plan, names: Mapping[str, str]) -> str:
    lines = [voice.PROPOSAL_HEADER.format(deadline=format_when(plan.deadline))]
    for i, s in enumerate(plan.candidate_slots, start=1):
        lines.append(
            voice.PROPOSAL_SLOT.format(index=i, venue=s.venue, when=format_when(s.starts_at))
        )
    lines.append(voice.PROPOSAL_FOOTER.format(stake=format_money(plan.stake_cents)))
    return "\n".join(lines)


def render_poll_closed(plan: Plan) -> str:
    slot = plan.locked_slot
    if slot is None:
        raise PlanError("poll is still open")
    body = voice.POLL_CLOSED.format(
        venue=slot.venue, when=format_when(slot.starts_at), stake=format_money(plan.stake_cents)
    )
    if tied(plan):
        body = f"{body}\n{voice.POLL_CLOSED_TIE}"
    return body


def render_lock(plan: Plan, names: Mapping[str, str]) -> str:
    slot = plan.locked_slot
    if slot is None or plan.outcome != "locked":
        raise PlanError("plan is not locked")
    who = [display_name(u, names) for u in plan.paid()]
    return "\n".join(
        [
            voice.LOCK_HEADER.format(venue=slot.venue, when=format_when(slot.starts_at)),
            voice.LOCK_WHO.format(names=join_names(who)),
            voice.LOCK_FOOTER,
        ]
    )


def render_receipt(
    plan: Plan,
    names: Mapping[str, str],
    attended: Sequence[str],
    forfeited: Sequence[str],
    *,
    next_round: datetime | None = None,
) -> str:
    """The post-dinner recap sent to the group. Built to be screenshotted (product-plan §7)."""
    slot = plan.locked_slot
    if slot is None:
        raise PlanError("plan has no locked slot")
    lines = [
        voice.RECEIPT_HEADER.format(venue=slot.venue, when=format_when(slot.starts_at)),
        voice.RECEIPT_ATTENDED.format(names=join_names([display_name(u, names) for u in attended])),
    ]
    if forfeited:
        share = (plan.stake_cents * len(forfeited)) // max(len(attended), 1)
        lines.append(
            voice.RECEIPT_FORFEITED.format(
                names=join_names([display_name(u, names) for u in forfeited]),
                share=format_money(share),
            )
        )
    else:
        lines.append(voice.RECEIPT_ALL_SHOWED.format(count=len(attended)))
    if next_round is not None:
        lines.append(voice.RECEIPT_FOOTER.format(next_round=format_when(next_round)))
    return "\n".join(lines)


# --- formatting helpers ------------------------------------------------------------------------


def display_name(user_id: str, names: Mapping[str, str]) -> str:
    return names.get(user_id, user_id)


def join_names(names: Sequence[str]) -> str:
    if not names:
        return "nobody"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def format_when(dt: datetime) -> str:
    """``Thu Sep 17, 7:30pm`` in the datetime's own timezone. Portable (no ``%-d``)."""
    hour = dt.hour % 12 or 12
    minute = f":{dt.minute:02d}" if dt.minute else ""
    suffix = "am" if dt.hour < 12 else "pm"
    return f"{dt:%a} {dt:%b} {dt.day}, {hour}{minute}{suffix}"


def format_money(cents: int) -> str:
    if cents % 100 == 0:
        return f"${cents // 100}"
    return f"${cents / 100:.2f}"
