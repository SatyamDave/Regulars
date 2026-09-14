"""What the bot needs from persistence, as a Protocol, plus an in-memory implementation.

``regulars.db.repo`` is adapted to this Protocol outside this package (the bot never imports
``regulars.db``). ``write_audit`` mirrors ``repo.write_audit``'s keyword signature so the adapter
is a pass-through. Everything here is synchronous: the repository is SQLAlchemy Core over a
``Connection`` and the async boundary lives in the transport, not the store.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from regulars.bot.events import DirectedRating
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Plan


@dataclass(frozen=True)
class Member:
    user_id: str
    handle: str
    name: str


@dataclass(frozen=True)
class GroupChat:
    """A group and the chat that renders it. ``members`` is the current roster."""

    group_id: str
    chat_id: str
    members: tuple[Member, ...]
    last_message_at_ms: int | None = None
    last_sender: str | None = None

    def names(self) -> dict[str, str]:
        return {m.user_id: m.name for m in self.members}

    def member(self, user_id: str) -> Member | None:
        return next((m for m in self.members if m.user_id == user_id), None)

    def member_by_handle(self, handle: str) -> Member | None:
        return next((m for m in self.members if m.handle == handle), None)


@dataclass(frozen=True)
class MorningAfterSession:
    """An attendee we have DMed and are waiting to hear back from. ``others`` is in prompt order."""

    user_id: str
    plan_id: str
    group_id: str
    others: tuple[str, ...]


@dataclass(frozen=True)
class AuditEntry:
    """Mirror of ``repo.AuditEntry``; every bot decision writes one (§9)."""

    id: str
    actor: str
    decision_type: str
    reason: str
    payload: dict[str, Any]
    created_at: datetime
    subject_user_id: str | None = None
    group_id: str | None = None
    plan_id: str | None = None
    persona_version_id: str | None = None
    judge_version: str | None = None


class BotStore(Protocol):
    # identity: iMessage handle <-> user id
    def user_for_handle(self, handle: str) -> str | None: ...
    def handle_for_user(self, user_id: str) -> str | None: ...
    def ensure_user(self, handle: str) -> str:
        """Return the user for ``handle``, creating one on first contact."""
        ...

    # groups
    def save_group(self, group: GroupChat) -> None: ...
    def get_group(self, group_id: str) -> GroupChat | None: ...
    def group_for_chat(self, chat_id: str) -> GroupChat | None: ...

    # plans
    def save_plan(self, plan: Plan) -> None: ...
    def get_plan(self, plan_id: str) -> Plan | None: ...
    def open_plans(self) -> list[Plan]: ...
    def open_plan_for_group(self, group_id: str) -> Plan | None: ...

    # onboarding
    def get_onboarding(self, user_id: str) -> OnboardingState | None: ...
    def save_onboarding(self, state: OnboardingState) -> None: ...

    # morning after
    def get_morning_after(self, user_id: str) -> MorningAfterSession | None: ...
    def save_morning_after(self, session: MorningAfterSession) -> None: ...
    def clear_morning_after(self, user_id: str) -> None: ...
    def save_ratings(self, ratings: Sequence[DirectedRating], plan_id: str) -> None: ...

    # audit (§9)
    def write_audit(
        self,
        *,
        actor: str,
        decision_type: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        subject_user_id: str | None = None,
        group_id: str | None = None,
        plan_id: str | None = None,
        persona_version_id: str | None = None,
        judge_version: str | None = None,
    ) -> AuditEntry: ...
    def audit_for_user(self, user_id: str) -> list[AuditEntry]: ...


@dataclass
class InMemoryBotStore:
    """Dict-backed store for tests and local runs. ``now`` stamps audit rows."""

    now: Callable[[], datetime] | None = None
    handles: dict[str, str] = field(default_factory=dict)  # handle -> user_id
    groups: dict[str, GroupChat] = field(default_factory=dict)
    plans: dict[str, Plan] = field(default_factory=dict)
    onboarding: dict[str, OnboardingState] = field(default_factory=dict)
    morning_after: dict[str, MorningAfterSession] = field(default_factory=dict)
    ratings: list[tuple[str, DirectedRating]] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)

    def _timestamp(self) -> datetime:
        return self.now() if self.now is not None else datetime.now(UTC)

    # identity
    def user_for_handle(self, handle: str) -> str | None:
        return self.handles.get(handle)

    def handle_for_user(self, user_id: str) -> str | None:
        return next((h for h, u in self.handles.items() if u == user_id), None)

    def bind_handle(self, handle: str, user_id: str) -> None:
        self.handles[handle] = user_id

    def ensure_user(self, handle: str) -> str:
        user = self.handles.get(handle)
        if user is None:
            user = f"user-{len(self.handles) + 1}"
            self.handles[handle] = user
        return user

    # groups
    def save_group(self, group: GroupChat) -> None:
        self.groups[group.group_id] = group

    def get_group(self, group_id: str) -> GroupChat | None:
        return self.groups.get(group_id)

    def group_for_chat(self, chat_id: str) -> GroupChat | None:
        return next((g for g in self.groups.values() if g.chat_id == chat_id), None)

    # plans
    def save_plan(self, plan: Plan) -> None:
        self.plans[plan.id] = plan

    def get_plan(self, plan_id: str) -> Plan | None:
        return self.plans.get(plan_id)

    def open_plans(self) -> list[Plan]:
        return [p for p in self.plans.values() if p.outcome == "open"]

    def open_plan_for_group(self, group_id: str) -> Plan | None:
        return next((p for p in self.open_plans() if p.group_id == group_id), None)

    # onboarding
    def get_onboarding(self, user_id: str) -> OnboardingState | None:
        return self.onboarding.get(user_id)

    def save_onboarding(self, state: OnboardingState) -> None:
        self.onboarding[state.user_id] = state

    # morning after
    def get_morning_after(self, user_id: str) -> MorningAfterSession | None:
        return self.morning_after.get(user_id)

    def save_morning_after(self, session: MorningAfterSession) -> None:
        self.morning_after[session.user_id] = session

    def clear_morning_after(self, user_id: str) -> None:
        self.morning_after.pop(user_id, None)

    def save_ratings(self, ratings: Sequence[DirectedRating], plan_id: str) -> None:
        self.ratings.extend((plan_id, r) for r in ratings)

    # audit
    def write_audit(
        self,
        *,
        actor: str,
        decision_type: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        subject_user_id: str | None = None,
        group_id: str | None = None,
        plan_id: str | None = None,
        persona_version_id: str | None = None,
        judge_version: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            id=f"audit-{len(self.audit) + 1}",
            actor=actor,
            decision_type=decision_type,
            reason=reason,
            payload=dict(payload or {}),
            created_at=self._timestamp(),
            subject_user_id=subject_user_id,
            group_id=group_id,
            plan_id=plan_id,
            persona_version_id=persona_version_id,
            judge_version=judge_version,
        )
        self.audit.append(entry)
        return entry

    def audit_for_user(self, user_id: str) -> list[AuditEntry]:
        return [a for a in self.audit if a.subject_user_id == user_id]
