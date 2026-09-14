"""The three flows that compose the five tools: open a group, lock a plan, run the morning after.

- :class:`GroupOpener` — one intro with a hook per person, then immediately the proposal.
  Never "get to know each other" (product-plan §3).
- :class:`LockFlow` — ``tick(now)`` closes polls past deadline, requests deposits, drops unpaid
  members 24h later (emitting :class:`BackfillRequest`), and locks once everyone left has paid.
- :class:`MorningAfter` — the receipt to the group, then a private directed prompt to each
  attendee. Replies parse into directed rows, one per (rater, ratee), never collapsed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from regulars.bot import voice
from regulars.bot.clock import Clock
from regulars.bot.events import BackfillRequest, DirectedRating
from regulars.bot.plan import (
    POLL_WINDOW,
    Plan,
    Slot,
    cancel,
    close_poll,
    display_name,
    drop_unpaid,
    format_money,
    format_when,
    lock,
    mark_paid,
    render_lock,
    render_poll_closed,
    render_proposal,
    render_receipt,
)
from regulars.bot.store import BotStore, GroupChat, Member, MorningAfterSession
from regulars.bot.tools import BotTools
from regulars.bot.transport import Transport


class FlowError(RuntimeError):
    """A flow was asked to act on state the store does not have."""


# --- group opening -----------------------------------------------------------------------------


def render_intro(members: Sequence[Member], hooks: Mapping[str, str]) -> str:
    lines = [voice.INTRO_HEADER.format(count=len(members))]
    for m in members:
        hook = hooks.get(m.user_id)
        if not hook:
            raise FlowError(f"no hook for {m.user_id!r}; every member gets one")
        lines.append(voice.INTRO_LINE.format(name=m.name, hook=hook))
    lines.append(voice.INTRO_FOOTER)
    return "\n".join(lines)


class GroupOpener:
    def __init__(
        self, tools: BotTools, transport: Transport, store: BotStore, clock: Clock
    ) -> None:
        self.tools = tools
        self.transport = transport
        self.store = store
        self.clock = clock

    async def open(
        self,
        group_id: str,
        members: Sequence[Member],
        hooks: Mapping[str, str],
        slots: Sequence[Slot],
        *,
        deadline: datetime | None = None,
        chat_name: str = "Regulars",
    ) -> Plan:
        intro = render_intro(members, hooks)  # validate hooks before touching the transport
        chat_id = await self.transport.create_group_chat([m.handle for m in members], chat_name)
        group = GroupChat(group_id=group_id, chat_id=chat_id, members=tuple(members))
        self.store.save_group(group)
        self.store.write_audit(
            actor="bot",
            decision_type="group_opened",
            reason=f"{len(members)} members, chat {chat_id}",
            payload={"members": [m.user_id for m in members], "hooks": dict(hooks)},
            group_id=group_id,
        )
        await self.tools.send_message(group_id, intro)
        plan = await self.tools.create_plan(
            group_id, slots, deadline or self.clock.now() + POLL_WINDOW
        )
        await self.tools.send_message(group_id, render_proposal(plan, group.names()))
        return plan


# --- lock --------------------------------------------------------------------------------------


class LockFlow:
    def __init__(self, tools: BotTools, store: BotStore, clock: Clock) -> None:
        self.tools = tools
        self.store = store
        self.clock = clock

    async def tick(self, now: datetime) -> list[BackfillRequest]:
        backfills: list[BackfillRequest] = []
        for plan in self.store.open_plans():
            if plan.poll_open and now >= plan.deadline:
                plan = await self._close_and_request_deposits(plan, now)
            plan, dropped = drop_unpaid(plan, now)
            if dropped:
                self.store.save_plan(plan)
                backfills.append(await self._drop(plan, dropped))
            if plan.outcome == "open" and plan.locked_slot_id is not None:
                if plan.all_paid():
                    await self._lock(plan, now)
                elif not plan.members():
                    self.store.save_plan(cancel(plan))
                    self.store.write_audit(
                        actor="bot",
                        decision_type="plan_cancelled",
                        reason="every member was dropped for non-payment",
                        group_id=plan.group_id,
                        plan_id=plan.id,
                    )
        return backfills

    async def record_payment(self, plan_id: str, user_id: str) -> Plan:
        """Payments webhook says the hold succeeded. Marks paid; the next tick may lock."""
        plan = self._plan(plan_id)
        plan = mark_paid(plan, user_id)
        self.store.save_plan(plan)
        await self.tools.log_signal(
            user_id, "deposit_paid", float(plan.stake_cents), group_id=plan.group_id
        )
        return plan

    async def _close_and_request_deposits(self, plan: Plan, now: datetime) -> Plan:
        plan = close_poll(plan, now)
        self.store.save_plan(plan)
        slot = plan.locked_slot
        if slot is None:
            raise RuntimeError(f"close_poll left plan {plan.id} without a winning slot")
        await self.tools.send_message(plan.group_id, render_poll_closed(plan))
        for user in plan.members():
            request = await self.tools.collect_deposit(plan.id, user, plan.stake_cents)
            when = format_when(slot.starts_at)
            stake = format_money(plan.stake_cents)
            if request.amount_after_credit_cents == 0:
                self.store.save_plan(mark_paid(self._plan(plan.id), user))
                body = voice.DEPOSIT_DM_COVERED.format(venue=slot.venue, when=when, stake=stake)
            elif request.amount_after_credit_cents < plan.stake_cents:
                body = voice.DEPOSIT_DM_CREDIT.format(
                    venue=slot.venue,
                    when=when,
                    stake=stake,
                    credit=format_money(plan.stake_cents - request.amount_after_credit_cents),
                    amount=format_money(request.amount_after_credit_cents),
                    link=request.pay_url_or_secret,
                )
            else:
                body = voice.DEPOSIT_DM.format(
                    venue=slot.venue,
                    when=when,
                    amount=format_money(request.amount_after_credit_cents),
                    link=request.pay_url_or_secret,
                )
            await self.tools.send_message(plan.group_id, body, to_user=user)
        return self._plan(plan.id)

    async def _drop(self, plan: Plan, dropped: Sequence[str]) -> BackfillRequest:
        slot = plan.locked_slot
        if slot is None:
            raise RuntimeError(f"cannot drop members from plan {plan.id}: no slot chosen")
        for user in dropped:
            await self.tools.send_message(
                plan.group_id, voice.DROPPED_DM.format(venue=slot.venue), to_user=user
            )
            await self.tools.remove_member(plan.group_id, user, voice.DROPPED_REASON)
        return BackfillRequest(
            group_id=plan.group_id, plan_id=plan.id, seats=len(dropped), dropped=tuple(dropped)
        )

    async def _lock(self, plan: Plan, now: datetime) -> Plan:
        plan = lock(plan, now)
        self.store.save_plan(plan)
        group = self.store.get_group(plan.group_id)
        names = group.names() if group is not None else {}
        await self.tools.send_message(plan.group_id, render_lock(plan, names))
        return plan

    def _plan(self, plan_id: str) -> Plan:
        plan = self.store.get_plan(plan_id)
        if plan is None:
            raise FlowError(f"unknown plan {plan_id!r}")
        return plan


# --- morning after -----------------------------------------------------------------------------

_LABELS: dict[str, tuple[str, ...]] = {
    "again": ("again", "see again", "1"),
    "off": ("off", "2"),
    "worth": ("worth", "3"),
}
_NONE_WORDS = frozenset({"none", "nobody", "no one", "noone", "no", "nah", "-", "nope", "n/a"})
_ALL_WORDS = frozenset({"everyone", "everybody", "all", "all of them"})
_SPLIT = re.compile(r"\s*(?:,|;|/|&|\band\b)\s*", re.IGNORECASE)


def render_morning_after(others: Sequence[str], names: Mapping[str, str]) -> str:
    lines = [voice.MORNING_AFTER_HEADER, voice.MORNING_AFTER_AGAIN]
    for i, u in enumerate(others, start=1):
        lines.append(voice.MORNING_AFTER_OTHER.format(index=i, name=display_name(u, names)))
    lines.extend([voice.MORNING_AFTER_OFF, voice.MORNING_AFTER_WORTH, voice.MORNING_AFTER_FORMAT])
    return "\n".join(lines)


class MorningAfter:
    def __init__(self, tools: BotTools, store: BotStore, clock: Clock) -> None:
        self.tools = tools
        self.store = store
        self.clock = clock

    async def start(
        self,
        plan: Plan,
        attended: Sequence[str],
        forfeited: Sequence[str] = (),
        *,
        next_round: datetime | None = None,
    ) -> None:
        group = self.store.get_group(plan.group_id)
        if group is None:
            raise FlowError(f"unknown group {plan.group_id!r}")
        names = group.names()
        await self.tools.send_message(
            plan.group_id, render_receipt(plan, names, attended, forfeited, next_round=next_round)
        )
        for user in attended:
            others = tuple(u for u in attended if u != user)
            self.store.save_morning_after(
                MorningAfterSession(
                    user_id=user, plan_id=plan.id, group_id=plan.group_id, others=others
                )
            )
            await self.tools.send_message(
                plan.group_id, render_morning_after(others, names), to_user=user
            )

    @staticmethod
    def parse_reply(user: str, text: str, others: Mapping[str, str]) -> list[DirectedRating]:
        """``others`` maps user_id -> display name, in the order they were numbered in the prompt.

        Returns one directed row per other attendee, or an empty list when the reply could not
        be read at all (no "again" line and not three positional lines).
        """
        fields = _split_fields(text)
        if "again" not in fields:
            return []
        again = _resolve_people(fields["again"], others)
        off = _resolve_people(fields.get("off", ""), others)
        worth = _worth(fields.get("worth", ""))
        return [
            DirectedRating(
                rater=user,
                ratee=ratee,
                would_see_again=ratee in again,
                felt_connected_1_7=worth,
                was_off=ratee in off,
            )
            for ratee in others
        ]


_LABELLED = re.compile(
    r"^(?:(see again|again|off|worth)\s*[:.)\-]?|([123])\s*[.):\-])\s*(.*)$", re.IGNORECASE
)


def _split_fields(text: str) -> dict[str, str]:
    """Labelled lines (``again:``, ``2)`` ...) or, failing that, three positional lines.

    A bare ``3`` is a value, not a label: numeric labels need a separator (``3.``, ``3)``).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fields: dict[str, str] = {}
    for line in lines:
        m = _LABELLED.match(line)
        if not m:
            continue
        label = (m.group(1) or m.group(2)).lower()
        for key, aliases in _LABELS.items():
            if label in aliases:
                fields[key] = m.group(3).strip()
    if "again" not in fields and len(lines) == 3:
        fields = {"again": lines[0], "off": lines[1], "worth": lines[2]}
    return fields


def _resolve_people(raw: str, others: Mapping[str, str]) -> set[str]:
    cleaned = raw.strip().strip(".").lower()
    if not cleaned or cleaned in _NONE_WORDS:
        return set()
    if cleaned in _ALL_WORDS:
        return set(others)
    ordered = list(others)
    found: set[str] = set()
    for chunk in _SPLIT.split(cleaned):
        chunk = chunk.strip()
        if not chunk:
            continue
        hit = _match_person(chunk, ordered, others)
        if hit is not None:
            found.add(hit)
            continue
        for token in chunk.split():
            hit = _match_person(token, ordered, others)
            if hit is not None:
                found.add(hit)
    return found


def _match_person(token: str, ordered: Sequence[str], others: Mapping[str, str]) -> str | None:
    token = token.strip(".,;:()#").lower()
    if not token:
        return None
    if token.isdigit():
        idx = int(token)
        return ordered[idx - 1] if 1 <= idx <= len(ordered) else None
    for user_id in ordered:
        name = others[user_id].lower()
        if token == name or name.startswith(token) or token.startswith(name):
            return user_id
    return None


def _worth(raw: str) -> int | None:
    m = re.search(r"\b([1-7])\b", raw)
    return int(m.group(1)) if m else None
