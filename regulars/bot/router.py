"""Router: one inbound message -> the actions the runtime should take.

DMs go to onboarding, the morning-after interview, the two standing commands ("what do you
know about me" -> :class:`LegibilityRequest`; "why was I removed" -> the latest
``member_removed`` audit reason), or a short fallback. Group messages are never answered;
they produce signals (reply latency, tapbacks) and, while a poll is open, votes.

The router reads and writes conversational state through the store synchronously and returns
:data:`OutboundAction` values for anything that needs a tool (sending, logging a signal) or
that another subpackage consumes (events).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

from regulars.bot import voice
from regulars.bot.clock import Clock
from regulars.bot.events import BotEvent, LegibilityRequest, PruneRequest
from regulars.bot.flows import MorningAfter
from regulars.bot.onboarding import Onboarding
from regulars.bot.plan import Plan, PlanError, format_when, vote
from regulars.bot.store import BotStore, GroupChat
from regulars.bot.transport import InboundMessage
from regulars.core.audit import latest_removal


@dataclass(frozen=True)
class SendMessage:
    body: str
    group_id: str | None = None
    to_user: str | None = None


@dataclass(frozen=True)
class LogSignal:
    user_id: str
    signal_type: str
    value: float
    group_id: str | None = None


@dataclass(frozen=True)
class Emit:
    event: BotEvent


OutboundAction = SendMessage | LogSignal | Emit

_LEGIBILITY = re.compile(r"what do you (know|have) (about|on) me", re.IGNORECASE)
_REMOVAL = re.compile(
    r"why (was|am) i (removed|dropped|kicked)|why did (you|i get) (remove|drop)", re.IGNORECASE
)
_BARE_VOTE = re.compile(r"^\s*(?:option|#)?\s*([1-9]|[a-f])\s*[.)]?\s*$", re.IGNORECASE)
_LEADING_VOTE = re.compile(
    r"^\s*(?:option|#)?\s*([1-9]|[a-f])\b\s+(?:works|for me|pls|please|is good|sounds good)",
    re.IGNORECASE,
)


class Router:
    def __init__(self, store: BotStore, onboarding: Onboarding, clock: Clock) -> None:
        self.store = store
        self.onboarding = onboarding
        self.clock = clock

    def handle(self, inbound: InboundMessage) -> list[OutboundAction]:
        group = self.store.group_for_chat(inbound.chat_id)
        if group is not None:
            return self._group_message(group, inbound)
        return self._dm(inbound)

    # --- group chat ----------------------------------------------------------------------------

    def _group_message(self, group: GroupChat, inbound: InboundMessage) -> list[OutboundAction]:
        member = group.member_by_handle(inbound.sender_handle)
        if member is None:
            return []
        user = member.user_id
        actions: list[OutboundAction] = []

        if inbound.is_tapback:
            actions.append(LogSignal(user, "tapback", 1.0, group_id=group.group_id))
            return actions

        if group.last_message_at_ms is not None and group.last_sender != user:
            latency = max(inbound.sent_at_ms - group.last_message_at_ms, 0)
            actions.append(LogSignal(user, "reply_latency_ms", float(latency), group.group_id))
        self.store.save_group(
            GroupChat(
                group_id=group.group_id,
                chat_id=group.chat_id,
                members=group.members,
                last_message_at_ms=inbound.sent_at_ms,
                last_sender=user,
            )
        )

        plan = self.store.open_plan_for_group(group.group_id)
        if plan is not None and plan.poll_open:
            slot_id = parse_vote(inbound.body, plan)
            if slot_id is not None:
                # A dropped member's vote raises PlanError and does not count.
                with contextlib.suppress(PlanError):
                    self.store.save_plan(vote(plan, user, slot_id))
        return actions

    # --- DMs -----------------------------------------------------------------------------------

    def _dm(self, inbound: InboundMessage) -> list[OutboundAction]:
        user = self.store.ensure_user(inbound.sender_handle)
        state = self.store.get_onboarding(user)
        if state is None or not state.done:
            step = (
                self.onboarding.start(user)
                if state is None
                else self.onboarding.step(state, inbound)
            )
            self.store.save_onboarding(step.state)
            actions: list[OutboundAction] = [SendMessage(r, to_user=user) for r in step.replies]
            actions.extend(Emit(e) for e in step.events)
            return actions

        session = self.store.get_morning_after(user)
        if session is not None:
            group = self.store.get_group(session.group_id)
            names = group.names() if group is not None else {}
            others = {u: names.get(u, u) for u in session.others}
            ratings = MorningAfter.parse_reply(user, inbound.body, others)
            if not ratings and others:
                return [SendMessage(voice.MORNING_AFTER_UNCLEAR, to_user=user)]
            self.store.save_ratings(ratings, session.plan_id)
            self.store.clear_morning_after(user)
            return [
                Emit(
                    PruneRequest(
                        group_id=session.group_id,
                        plan_id=session.plan_id,
                        rater=user,
                        ratings=tuple(ratings),
                    )
                ),
                SendMessage(voice.MORNING_AFTER_THANKS, to_user=user),
            ]

        if _LEGIBILITY.search(inbound.body):
            return [
                SendMessage(voice.LEGIBILITY_ACK, to_user=user),
                Emit(LegibilityRequest(user_id=user)),
            ]
        if _REMOVAL.search(inbound.body):
            latest = latest_removal(self.store.audit_for_user(user))
            if latest is None:
                return [SendMessage(voice.NOT_REMOVED, to_user=user)]
            return [
                SendMessage(
                    voice.REMOVAL_REASON.format(
                        when=format_when(latest.created_at), reason=latest.reason
                    ),
                    to_user=user,
                )
            ]
        return [SendMessage(voice.DM_FALLBACK, to_user=user)]


def parse_vote(text: str, plan: Plan) -> str | None:
    """``"1"``, ``"2."``, ``"A"``, ``"option 2"``, ``"2 works"``, or a venue name -> slot id."""
    slots = plan.candidate_slots
    m = _BARE_VOTE.match(text) or _LEADING_VOTE.match(text)
    if m:
        token = m.group(1).lower()
        idx = int(token) if token.isdigit() else ord(token) - ord("a") + 1
        return slots[idx - 1].id if 1 <= idx <= len(slots) else None
    lowered = text.lower()
    hits = [s for s in slots if s.venue.lower() in lowered]
    if len(hits) == 1:
        return hits[0].id
    return None
