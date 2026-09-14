"""``SqlBotStore``: ``regulars.bot.store.BotStore`` over ``regulars.db.repo`` (0002 tables).

One transaction per method, as the module docstring on ``BotStore`` asks for: the bot's store
is synchronous, so each call opens and closes its own ``engine.begin()`` block. The roster for a
``GroupChat`` is reconstituted from ``memberships`` (active ones) joined with ``handles``, rather
than duplicated here, so the group's membership history stays the single source of truth (§1).

``ensure_user`` creates a placeholder ``users`` row on first contact: real signup does not yet
know the person's zip or a hashed phone number at the moment they text the bot, so those fields
get placeholders (``phone_hash=handle``, ``zip="00000"``) that a later onboarding step is
expected to correct. This is a integration seam, not a §1 schema change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, Engine, select

from regulars.bot.events import DirectedRating
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Plan
from regulars.bot.store import AuditEntry, GroupChat, Member, MorningAfterSession
from regulars.db import repo, schema

PLACEHOLDER_ZIP = "00000"


def _now() -> datetime:
    return datetime.now(UTC)


class SqlBotStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # --- identity --------------------------------------------------------------------------

    def user_for_handle(self, handle: str) -> str | None:
        with self.engine.connect() as conn:
            row = repo.get_handle(conn, handle)
            return row.user_id if row is not None else None

    def handle_for_user(self, user_id: str) -> str | None:
        with self.engine.connect() as conn:
            row = repo.handle_for_user(conn, user_id)
            return row.handle if row is not None else None

    def ensure_user(self, handle: str) -> str:
        with self.engine.begin() as conn:
            existing = repo.get_handle(conn, handle)
            if existing is not None:
                return existing.user_id
            user = repo.create_user(conn, phone_hash=handle, zip_code=PLACEHOLDER_ZIP)
            repo.bind_handle(conn, handle=handle, user_id=user.id)
            return user.id

    # --- groups ------------------------------------------------------------------------------

    def save_group(self, group: GroupChat) -> None:
        with self.engine.begin() as conn:
            repo.save_group_chat(
                conn,
                group_id=group.group_id,
                chat_id=group.chat_id,
                last_message_at_ms=group.last_message_at_ms,
                last_sender=group.last_sender,
            )
            m = schema.memberships
            current_round = conn.execute(
                select(m.c.joined_round)
                .where(m.c.group_id == group.group_id)
                .order_by(m.c.joined_round.desc())
                .limit(1)
            ).scalar()
            round_no = 1 if current_round is None else int(current_round)
            active = {
                mm.user_id
                for mm in repo.memberships_for_group(conn, group.group_id, active_only=True)
            }
            wanted = {mem.user_id for mem in group.members}
            for user_id in wanted - active:
                repo.add_membership(conn, group.group_id, user_id, round_no)
            for user_id in active - wanted:
                repo.end_membership(
                    conn, group.group_id, user_id, left_round=round_no, exit_reason="removed"
                )
            for member in group.members:
                repo.bind_handle(
                    conn, handle=member.handle, user_id=member.user_id, display_name=member.name
                )

    def get_group(self, group_id: str) -> GroupChat | None:
        with self.engine.connect() as conn:
            row = repo.get_group_chat(conn, group_id)
            if row is None:
                return None
            return self._group_chat(conn, row)

    def group_for_chat(self, chat_id: str) -> GroupChat | None:
        with self.engine.connect() as conn:
            row = repo.group_chat_for_chat_id(conn, chat_id)
            if row is None:
                return None
            return self._group_chat(conn, row)

    def _group_chat(self, conn: Connection, row: repo.GroupChatRow) -> GroupChat:
        members = tuple(
            Member(
                user_id=mm.user_id,
                handle=h.handle if h else mm.user_id,
                name=h.display_name if h else mm.user_id,
            )
            for mm in repo.memberships_for_group(conn, row.group_id, active_only=True)
            for h in (repo.handle_for_user(conn, mm.user_id),)
        )
        return GroupChat(
            group_id=row.group_id,
            chat_id=row.chat_id,
            members=members,
            last_message_at_ms=row.last_message_at_ms,
            last_sender=row.last_sender,
        )

    # --- plans -------------------------------------------------------------------------------

    def save_plan(self, plan: Plan) -> None:
        """Store the full record, and mirror it onto the §1 ``plans``/``rsvps`` tables so other
        subpackages (e.g. ``regulars.pipeline.post_dinner``, which reads ``rsvps.state``) see the
        bot's plan lifecycle without knowing about ``plan_records``."""
        with self.engine.begin() as conn:
            repo.ensure_plan_row(conn, plan.id, plan.group_id)
            repo.save_plan_record(conn, plan.id, plan.model_dump(mode="json"))
            if plan.locked_slot_id is not None and plan.outcome == "locked":
                row = repo.get_plan(conn, plan.id)
                if row.locked_at is None:
                    slot = plan.slot(plan.locked_slot_id)
                    repo.lock_plan(conn, plan.id, venue=slot.venue, starts_at=slot.starts_at)
            for user_id, state in plan.commitment.items():
                repo.upsert_rsvp(
                    conn,
                    plan.id,
                    user_id,
                    state=state,
                    deposit_intent_id=plan.deposit_intents.get(user_id),
                    paid_at=_now() if state == "paid" else None,
                )

    def get_plan(self, plan_id: str) -> Plan | None:
        with self.engine.connect() as conn:
            row = repo.get_plan_record(conn, plan_id)
            return None if row is None else Plan.model_validate(row.plan_json)

    def open_plans(self) -> list[Plan]:
        with self.engine.connect() as conn:
            return [Plan.model_validate(r.plan_json) for r in repo.open_plan_records(conn)]

    def open_plan_for_group(self, group_id: str) -> Plan | None:
        return next((p for p in self.open_plans() if p.group_id == group_id), None)

    # --- onboarding ----------------------------------------------------------------------------

    def get_onboarding(self, user_id: str) -> OnboardingState | None:
        with self.engine.connect() as conn:
            row = repo.get_onboarding_state(conn, user_id)
            return None if row is None else OnboardingState.model_validate(row.state_json)

    def save_onboarding(self, state: OnboardingState) -> None:
        with self.engine.begin() as conn:
            repo.save_onboarding_state(conn, state.user_id, state.model_dump(mode="json"))

    # --- morning after -------------------------------------------------------------------------

    def get_morning_after(self, user_id: str) -> MorningAfterSession | None:
        with self.engine.connect() as conn:
            row = repo.get_morning_after_state(conn, user_id)
            if row is None:
                return None
            return MorningAfterSession(
                user_id=row.user_id,
                plan_id=row.plan_id,
                group_id=row.group_id,
                others=tuple(row.others_json),
            )

    def save_morning_after(self, session: MorningAfterSession) -> None:
        with self.engine.begin() as conn:
            repo.save_morning_after_state(
                conn,
                user_id=session.user_id,
                plan_id=session.plan_id,
                group_id=session.group_id,
                others=session.others,
            )

    def clear_morning_after(self, user_id: str) -> None:
        with self.engine.begin() as conn:
            repo.clear_morning_after_state(conn, user_id)

    def save_ratings(self, ratings: Sequence[DirectedRating], plan_id: str) -> None:
        with self.engine.begin() as conn:
            for r in ratings:
                repo.insert_rating(
                    conn,
                    plan_id,
                    rater_id=r.rater,
                    ratee_id=r.ratee,
                    would_see_again=r.would_see_again,
                    felt_connected_1_7=r.felt_connected_1_7,
                )

    # --- audit -----------------------------------------------------------------------------

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
        with self.engine.begin() as conn:
            row = repo.write_audit(
                conn,
                actor=actor,
                decision_type=decision_type,
                reason=reason,
                payload=payload,
                subject_user_id=subject_user_id,
                group_id=group_id,
                plan_id=plan_id,
                persona_version_id=persona_version_id,
                judge_version=judge_version,
            )
            return _audit_entry(row)

    def audit_for_user(self, user_id: str) -> list[AuditEntry]:
        with self.engine.connect() as conn:
            return [_audit_entry(r) for r in repo.audit_for_user(conn, user_id)]


def _audit_entry(row: repo.AuditEntry) -> AuditEntry:
    return AuditEntry(
        id=row.id,
        actor=row.actor,
        decision_type=row.decision_type,
        reason=row.reason,
        payload=dict(row.payload),
        created_at=row.created_at,
        subject_user_id=row.subject_user_id,
        group_id=row.group_id,
        plan_id=row.plan_id,
        persona_version_id=row.persona_version_id,
        judge_version=row.judge_version,
    )
