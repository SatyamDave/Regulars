"""Data export (§10 Privacy architecture; GDPR Art. 15, CCPA right-to-know).

:func:`build_data_export` walks every table that can hold something about one user and returns a
single :class:`DataExport` — a portable, human-and-machine-readable record suitable for a
"download your data" button. :meth:`DataExport.to_json` serializes it.

**The one deliberate scope decision.** A user's export includes every message *they* sent, plus
metadata about groups and plans they participated in, but never another participant's message
body. ``regulars.db.repo.messages_by_user`` filters on ``user_id`` at the query level — it is not
possible to build an export that leaks someone else's chat content by construction. This mirrors
how ``ratings`` are already directed and never collapsed (§1 rule 2): the other person's side of a
shared interaction belongs to their own export, not to this one. Group and membership rows are
included because they describe *this* user's participation, not other members' behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Engine

from regulars.core.persona import PersonaSpec
from regulars.db import repo


class ExportedPersonaVersion(BaseModel, frozen=True):
    id: str
    version: int
    spec: PersonaSpec
    source: str
    created_at: datetime


class ExportedRatingGiven(BaseModel, frozen=True):
    plan_id: str
    ratee_id: str
    would_see_again: bool
    felt_connected_1_7: int | None
    submitted_at: datetime


class ExportedRatingReceived(BaseModel, frozen=True):
    plan_id: str
    rater_id: str
    would_see_again: bool
    felt_connected_1_7: int | None
    submitted_at: datetime


class ExportedMembership(BaseModel, frozen=True):
    group_id: str
    joined_round: int
    left_round: int | None
    exit_reason: str | None


class ExportedMessage(BaseModel, frozen=True):
    id: str
    group_id: str
    sent_at_ms: int
    body: str
    reply_to: str | None
    tapback_target: str | None
    char_count: int


class ExportedDepositEntry(BaseModel, frozen=True):
    id: str
    plan_id: str | None
    amount_cents: int
    kind: str
    created_at: datetime


class ExportedAuditEntry(BaseModel, frozen=True):
    id: str
    actor: str
    decision_type: str
    group_id: str | None
    plan_id: str | None
    persona_version_id: str | None
    judge_version: str | None
    reason: str
    payload: dict[str, Any]
    created_at: datetime


class DataExport(BaseModel, frozen=True):
    """Everything the system knows about one user, as of ``exported_at``.

    See the module docstring for the one deliberate exclusion: other participants' message
    bodies in shared group chats.
    """

    user_id: str
    exported_at: datetime
    zip: str
    status: str
    created_at: datetime
    persona_versions: list[ExportedPersonaVersion]
    ratings_given: list[ExportedRatingGiven]
    ratings_received: list[ExportedRatingReceived]
    memberships: list[ExportedMembership]
    messages: list[ExportedMessage]
    deposit_ledger: list[ExportedDepositEntry]
    audit_log: list[ExportedAuditEntry]

    def to_json(self) -> str:
        """A single portable JSON document — the "download your data" payload."""
        return self.model_dump_json(indent=2)


async def build_data_export(engine: Engine, user_id: str) -> DataExport:
    """Collect every row the system holds about ``user_id`` into one :class:`DataExport`.

    ``async`` for symmetry with the FastAPI route that calls it (and so a future move to an async
    database driver is a drop-in swap); the current repository layer is synchronous SQLAlchemy
    Core, so this runs to completion within one ``engine.begin()`` block with no actual await.
    Raises ``regulars.db.repo.NotFound`` if ``user_id`` does not exist.
    """
    with engine.begin() as conn:
        user = repo.get_user(conn, user_id)
        persona_versions = [
            ExportedPersonaVersion(
                id=pv.id,
                version=pv.version,
                spec=pv.spec,
                source=pv.source,
                created_at=pv.created_at,
            )
            for pv in repo.persona_versions_for_user(conn, user_id)
        ]
        ratings_given = [
            ExportedRatingGiven(
                plan_id=r.plan_id,
                ratee_id=r.ratee_id,
                would_see_again=r.would_see_again,
                felt_connected_1_7=r.felt_connected_1_7,
                submitted_at=r.submitted_at,
            )
            for r in repo.ratings_given_by(conn, user_id)
        ]
        ratings_received = [
            ExportedRatingReceived(
                plan_id=r.plan_id,
                rater_id=r.rater_id,
                would_see_again=r.would_see_again,
                felt_connected_1_7=r.felt_connected_1_7,
                submitted_at=r.submitted_at,
            )
            for r in repo.ratings_received_by(conn, user_id)
        ]
        memberships = [
            ExportedMembership(
                group_id=m.group_id,
                joined_round=m.joined_round,
                left_round=m.left_round,
                exit_reason=m.exit_reason,
            )
            for m in repo.memberships_for_user(conn, user_id)
        ]
        # repo.messages_by_user filters on user_id at the SQL level: this can only ever contain
        # messages this user authored, never another participant's body (see module docstring).
        messages = [
            ExportedMessage(
                id=m.id,
                group_id=m.group_id,
                sent_at_ms=m.sent_at_ms,
                body=m.body,
                reply_to=m.reply_to,
                tapback_target=m.tapback_target,
                char_count=m.char_count,
            )
            for m in repo.messages_by_user(conn, user_id)
        ]
        deposit_ledger = [
            ExportedDepositEntry(
                id=d.id,
                plan_id=d.plan_id,
                amount_cents=d.amount_cents,
                kind=d.kind,
                created_at=d.created_at,
            )
            for d in repo.deposit_credits_for_user(conn, user_id)
        ]
        audit_log = [
            ExportedAuditEntry(
                id=a.id,
                actor=a.actor,
                decision_type=a.decision_type,
                group_id=a.group_id,
                plan_id=a.plan_id,
                persona_version_id=a.persona_version_id,
                judge_version=a.judge_version,
                reason=a.reason,
                payload=a.payload,
                created_at=a.created_at,
            )
            for a in repo.audit_for_user(conn, user_id)
        ]
    return DataExport(
        user_id=user.id,
        exported_at=datetime.now(UTC),
        zip=user.zip,
        status=user.status,
        created_at=user.created_at,
        persona_versions=persona_versions,
        ratings_given=ratings_given,
        ratings_received=ratings_received,
        memberships=memberships,
        messages=messages,
        deposit_ledger=deposit_ledger,
        audit_log=audit_log,
    )
