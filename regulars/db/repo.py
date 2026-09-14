"""Thin typed repository functions over a SQLAlchemy ``Connection``.

Conventions:

- Every function takes the ``Connection`` first and never commits. Callers own the transaction::

      with engine.begin() as conn:
          user = repo.create_user(conn, phone_hash="...", zip_code="94110")

- Ids are UUID4 strings minted here. Timestamps are UTC-aware on the way in and out (SQLite drops
  the offset; we put it back on read).
- Functions return frozen pydantic records, never raw rows.
- Domain violations raise ``ValueError`` before touching the database; missing rows raise
  ``NotFound``. Constraint violations that slip past (races) surface as
  ``sqlalchemy.exc.IntegrityError``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection, Row, and_, func, insert, select, update

from regulars.core.features import FEATURE_NAMES
from regulars.core.persona import PersonaSource, PersonaSpec
from regulars.core.transcript import Transcript
from regulars.db import schema
from regulars.db.schema import (
    ASSEMBLY_METHODS,
    DEPOSIT_KINDS,
    HOLD_STATES,
    PERSONA_SOURCES,
    WEBHOOK_STATUSES,
    AssemblyMethod,
    DepositKind,
    HoldState,
    WebhookStatus,
)


class NotFound(LookupError):
    """The referenced row does not exist."""


# --- records -----------------------------------------------------------------------------------


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class User(_Record):
    id: str
    phone_hash: str
    zip: str
    created_at: datetime
    status: str
    exploration_flag: bool


class PersonaVersion(_Record):
    id: str
    user_id: str
    version: int
    spec: PersonaSpec
    source: PersonaSource
    created_at: datetime


class Sim(_Record):
    id: str
    user_a: str
    user_b: str
    persona_ver_a: str
    persona_ver_b: str
    scenario_id: str
    seed: int
    transcript: Transcript
    created_at: datetime


class SimFeature(_Record):
    sim_id: str
    feature_name: str
    value: float


class Prediction(_Record):
    id: str
    user_a: str
    user_b: str
    judge_version: str
    y_hat: float
    y_hat_calibrated: float | None
    created_at: datetime


class Group(_Record):
    id: str
    zip: str
    created_at: datetime
    assembly_method: AssemblyMethod
    status: str


class Membership(_Record):
    group_id: str
    user_id: str
    joined_round: int
    left_round: int | None
    exit_reason: str | None


class Plan(_Record):
    id: str
    group_id: str
    proposed_at: datetime
    locked_at: datetime | None
    venue: str | None
    starts_at: datetime | None


class Rsvp(_Record):
    plan_id: str
    user_id: str
    state: str
    deposit_intent_id: str | None
    paid_at: datetime | None


class Attendance(_Record):
    plan_id: str
    user_id: str
    attended: bool
    source: str


class Message(_Record):
    id: str
    group_id: str
    user_id: str
    sent_at_ms: int
    body: str
    reply_to: str | None
    tapback_target: str | None
    char_count: int


class Rating(_Record):
    plan_id: str
    rater_id: str
    ratee_id: str
    would_see_again: bool
    felt_connected_1_7: int | None
    submitted_at: datetime


class AuditEntry(_Record):
    id: str
    actor: str
    decision_type: str
    subject_user_id: str | None
    group_id: str | None
    plan_id: str | None
    persona_version_id: str | None
    judge_version: str | None
    reason: str
    payload: dict[str, Any]
    created_at: datetime


class DepositCredit(_Record):
    id: str
    user_id: str
    plan_id: str | None
    amount_cents: int
    kind: DepositKind
    created_at: datetime


class Handle(_Record):
    handle: str
    user_id: str
    display_name: str
    created_at: datetime


class GroupChatRow(_Record):
    group_id: str
    chat_id: str
    last_message_at_ms: int | None
    last_sender: str | None


class PlanRecord(_Record):
    plan_id: str
    plan_json: dict[str, Any]
    updated_at: datetime


class OnboardingStateRow(_Record):
    user_id: str
    state_json: dict[str, Any]
    updated_at: datetime


class MorningAfterStateRow(_Record):
    user_id: str
    plan_id: str
    group_id: str
    others_json: list[str]
    updated_at: datetime


class DepositHoldRow(_Record):
    plan_id: str
    user_id: str
    intent_id: str | None
    amount_cents: int
    credit_cents: int
    state: HoldState
    updated_at: datetime


class ProcessedWebhookEventRow(_Record):
    event_id: str
    event_type: str
    received_at: datetime
    processed_at: datetime | None
    status: WebhookStatus
    error: str | None


# --- helpers -----------------------------------------------------------------------------------


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything we store is UTC, so say so."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _row_dict(row: Row[Any]) -> dict[str, Any]:
    data = dict(row._mapping)
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = _utc(value)
    return data


def _one(conn: Connection, stmt: Any, what: str) -> dict[str, Any]:
    row = conn.execute(stmt).first()
    if row is None:
        raise NotFound(what)
    return _row_dict(row)


# --- users -------------------------------------------------------------------------------------


def create_user(
    conn: Connection,
    *,
    phone_hash: str,
    zip_code: str,
    status: str = "active",
    exploration_flag: bool = False,
    user_id: str | None = None,
) -> User:
    record = User(
        id=user_id or _new_id(),
        phone_hash=phone_hash,
        zip=zip_code,
        created_at=_now(),
        status=status,
        exploration_flag=exploration_flag,
    )
    conn.execute(insert(schema.users).values(**record.model_dump()))
    return record


def get_user(conn: Connection, user_id: str) -> User:
    return User(**_one(conn, select(schema.users).where(schema.users.c.id == user_id), "user"))


def list_users(
    conn: Connection, *, zip_code: str | None = None, status: str | None = None
) -> list[User]:
    u = schema.users
    stmt = select(u)
    if zip_code is not None:
        stmt = stmt.where(u.c.zip == zip_code)
    if status is not None:
        stmt = stmt.where(u.c.status == status)
    stmt = stmt.order_by(u.c.id)
    return [User(**_row_dict(r)) for r in conn.execute(stmt)]


# --- persona versions (append-only, §1 rule 1) -------------------------------------------------


def append_persona_version(
    conn: Connection, user_id: str, spec: PersonaSpec, source: PersonaSource
) -> PersonaVersion:
    """Append ``spec`` as the next version for ``user_id``.

    The version number is ``max(version) + 1`` computed inside the caller's transaction. Two
    concurrent appends for the same user cannot both succeed: the unique key on
    (user_id, version) makes the loser fail with ``IntegrityError`` rather than overwrite.
    Existing rows are never updated; there is deliberately no function that does so.
    """
    if source not in PERSONA_SOURCES:
        raise ValueError(f"unknown persona source {source!r}; expected one of {PERSONA_SOURCES}")
    pv = schema.persona_versions
    current = conn.execute(
        select(func.coalesce(func.max(pv.c.version), 0)).where(pv.c.user_id == user_id)
    ).scalar_one()
    record = PersonaVersion(
        id=_new_id(),
        user_id=user_id,
        version=int(current) + 1,
        spec=spec,
        source=source,
        created_at=_now(),
    )
    conn.execute(
        insert(pv).values(
            id=record.id,
            user_id=record.user_id,
            version=record.version,
            spec_json=spec.model_dump(mode="json"),
            source=record.source,
            created_at=record.created_at,
        )
    )
    return record


def _persona_from_row(data: dict[str, Any]) -> PersonaVersion:
    return PersonaVersion(
        id=data["id"],
        user_id=data["user_id"],
        version=data["version"],
        spec=PersonaSpec.model_validate(data["spec_json"]),
        source=data["source"],
        created_at=data["created_at"],
    )


def get_persona_version(conn: Connection, persona_version_id: str) -> PersonaVersion:
    pv = schema.persona_versions
    return _persona_from_row(
        _one(conn, select(pv).where(pv.c.id == persona_version_id), "persona_version")
    )


def latest_persona_version(conn: Connection, user_id: str) -> PersonaVersion | None:
    pv = schema.persona_versions
    row = conn.execute(
        select(pv).where(pv.c.user_id == user_id).order_by(pv.c.version.desc()).limit(1)
    ).first()
    return None if row is None else _persona_from_row(_row_dict(row))


# --- sims, features, predictions ---------------------------------------------------------------


def insert_sim(
    conn: Connection,
    *,
    user_a: str,
    user_b: str,
    persona_ver_a: str,
    persona_ver_b: str,
    scenario_id: str,
    seed: int,
    transcript: Transcript,
) -> Sim:
    record = Sim(
        id=_new_id(),
        user_a=user_a,
        user_b=user_b,
        persona_ver_a=persona_ver_a,
        persona_ver_b=persona_ver_b,
        scenario_id=scenario_id,
        seed=seed,
        transcript=transcript,
        created_at=_now(),
    )
    conn.execute(
        insert(schema.sims).values(
            id=record.id,
            user_a=user_a,
            user_b=user_b,
            persona_ver_a=persona_ver_a,
            persona_ver_b=persona_ver_b,
            scenario_id=scenario_id,
            seed=seed,
            transcript_json=transcript.model_dump(mode="json"),
            created_at=record.created_at,
        )
    )
    return record


def insert_sim_features(
    conn: Connection, sim_id: str, features: Mapping[str, float]
) -> list[SimFeature]:
    """Store extracted features for a sim. Keys must come from ``FEATURE_NAMES``."""
    unknown = sorted(set(features) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unknown feature names: {unknown}")
    records = [
        SimFeature(sim_id=sim_id, feature_name=name, value=float(value))
        for name, value in features.items()
    ]
    if records:
        conn.execute(insert(schema.sim_features), [r.model_dump() for r in records])
    return records


def sim_features_for(conn: Connection, sim_id: str) -> dict[str, float]:
    sf = schema.sim_features
    rows = conn.execute(select(sf.c.feature_name, sf.c.value).where(sf.c.sim_id == sim_id))
    return {name: value for name, value in rows}


def latest_predictions_for(
    conn: Connection, user_ids: Sequence[str], *, judge_version: str | None = None
) -> list[Prediction]:
    """The most recent prediction per unordered pair drawn from ``user_ids``, newest wins."""
    p = schema.predictions
    ids = set(user_ids)
    stmt = select(p)
    if judge_version is not None:
        stmt = stmt.where(p.c.judge_version == judge_version)
    stmt = stmt.order_by(p.c.created_at)
    latest: dict[tuple[str, str], Prediction] = {}
    for row in conn.execute(stmt):
        record = Prediction(**_row_dict(row))
        if record.user_a not in ids or record.user_b not in ids:
            continue
        a, b = sorted((record.user_a, record.user_b))
        latest[(a, b)] = record  # ascending created_at: last write per key wins
    return list(latest.values())


def insert_prediction(
    conn: Connection,
    *,
    user_a: str,
    user_b: str,
    judge_version: str,
    y_hat: float,
    y_hat_calibrated: float | None = None,
) -> Prediction:
    record = Prediction(
        id=_new_id(),
        user_a=user_a,
        user_b=user_b,
        judge_version=judge_version,
        y_hat=y_hat,
        y_hat_calibrated=y_hat_calibrated,
        created_at=_now(),
    )
    conn.execute(insert(schema.predictions).values(**record.model_dump()))
    return record


# --- groups and memberships --------------------------------------------------------------------


def create_group(
    conn: Connection,
    *,
    zip_code: str,
    assembly_method: AssemblyMethod,
    status: str = "active",
) -> Group:
    """Create a group. ``assembly_method`` is required: it is the exploration-arm label (§6)."""
    if assembly_method not in ASSEMBLY_METHODS:
        raise ValueError(
            f"assembly_method must be one of {ASSEMBLY_METHODS}, got {assembly_method!r}"
        )
    record = Group(
        id=_new_id(),
        zip=zip_code,
        created_at=_now(),
        assembly_method=assembly_method,
        status=status,
    )
    conn.execute(insert(schema.groups).values(**record.model_dump()))
    return record


def get_group(conn: Connection, group_id: str) -> Group:
    return Group(**_one(conn, select(schema.groups).where(schema.groups.c.id == group_id), "group"))


def add_membership(conn: Connection, group_id: str, user_id: str, joined_round: int) -> Membership:
    record = Membership(
        group_id=group_id,
        user_id=user_id,
        joined_round=joined_round,
        left_round=None,
        exit_reason=None,
    )
    conn.execute(insert(schema.memberships).values(**record.model_dump()))
    return record


def end_membership(
    conn: Connection, group_id: str, user_id: str, *, left_round: int, exit_reason: str
) -> Membership:
    """Close the user's open membership in the group (the one with ``left_round IS NULL``)."""
    m = schema.memberships
    open_row = and_(m.c.group_id == group_id, m.c.user_id == user_id, m.c.left_round.is_(None))
    result = conn.execute(
        update(m).where(open_row).values(left_round=left_round, exit_reason=exit_reason)
    )
    if result.rowcount != 1:
        raise NotFound("open membership")
    return Membership(
        **_one(
            conn,
            select(m).where(
                and_(m.c.group_id == group_id, m.c.user_id == user_id, m.c.left_round == left_round)
            ),
            "membership",
        )
    )


def memberships_for_group(
    conn: Connection, group_id: str, *, active_only: bool = False
) -> list[Membership]:
    m = schema.memberships
    stmt = select(m).where(m.c.group_id == group_id)
    if active_only:
        stmt = stmt.where(m.c.left_round.is_(None))
    stmt = stmt.order_by(m.c.joined_round, m.c.user_id)
    return [Membership(**_row_dict(r)) for r in conn.execute(stmt)]


# --- plans, rsvps, attendance ------------------------------------------------------------------


def create_plan(
    conn: Connection,
    group_id: str,
    *,
    venue: str | None = None,
    starts_at: datetime | None = None,
) -> Plan:
    record = Plan(
        id=_new_id(),
        group_id=group_id,
        proposed_at=_now(),
        locked_at=None,
        venue=venue,
        starts_at=starts_at,
    )
    conn.execute(insert(schema.plans).values(**record.model_dump()))
    return record


def ensure_plan_row(conn: Connection, plan_id: str, group_id: str) -> None:
    """Insert a ``plans`` row under a caller-chosen id if one doesn't exist yet.

    The bot mints its own plan ids (``regulars.bot.tools.BotTools.create_plan``) rather than
    calling :func:`create_plan`, so ``regulars.db.bot_store.SqlBotStore.save_plan`` uses this to
    keep the §1 ``plans`` table in sync without requiring a separate creation step.
    """
    p = schema.plans
    if conn.execute(select(p.c.id).where(p.c.id == plan_id)).first() is not None:
        return
    conn.execute(
        insert(p).values(
            id=plan_id,
            group_id=group_id,
            proposed_at=_now(),
            locked_at=None,
            venue=None,
            starts_at=None,
        )
    )


def get_plan(conn: Connection, plan_id: str) -> Plan:
    return Plan(**_one(conn, select(schema.plans).where(schema.plans.c.id == plan_id), "plan"))


def lock_plan(conn: Connection, plan_id: str, *, venue: str, starts_at: datetime) -> Plan:
    """Fix venue and time. Locking an already-locked plan is an error."""
    p = schema.plans
    result = conn.execute(
        update(p)
        .where(and_(p.c.id == plan_id, p.c.locked_at.is_(None)))
        .values(locked_at=_now(), venue=venue, starts_at=starts_at)
    )
    if result.rowcount != 1:
        if conn.execute(select(p.c.id).where(p.c.id == plan_id)).first() is None:
            raise NotFound("plan")
        raise ValueError(f"plan {plan_id} is already locked")
    return get_plan(conn, plan_id)


def upsert_rsvp(
    conn: Connection,
    plan_id: str,
    user_id: str,
    *,
    state: str,
    deposit_intent_id: str | None = None,
    paid_at: datetime | None = None,
) -> Rsvp:
    r = schema.rsvps
    key = and_(r.c.plan_id == plan_id, r.c.user_id == user_id)
    values = {"state": state, "deposit_intent_id": deposit_intent_id, "paid_at": paid_at}
    if conn.execute(update(r).where(key).values(**values)).rowcount == 0:
        conn.execute(insert(r).values(plan_id=plan_id, user_id=user_id, **values))
    return Rsvp(**_one(conn, select(r).where(key), "rsvp"))


def rsvps_for_plan(conn: Connection, plan_id: str) -> list[Rsvp]:
    r = schema.rsvps
    stmt = select(r).where(r.c.plan_id == plan_id).order_by(r.c.user_id)
    return [Rsvp(**_row_dict(row)) for row in conn.execute(stmt)]


def record_attendance(
    conn: Connection, plan_id: str, user_id: str, *, attended: bool, source: str
) -> Attendance:
    """Record (or correct) whether ``user_id`` showed up. Last write wins."""
    a = schema.attendance
    key = and_(a.c.plan_id == plan_id, a.c.user_id == user_id)
    values = {"attended": attended, "source": source}
    if conn.execute(update(a).where(key).values(**values)).rowcount == 0:
        conn.execute(insert(a).values(plan_id=plan_id, user_id=user_id, **values))
    return Attendance(plan_id=plan_id, user_id=user_id, attended=attended, source=source)


def attendance_for_plan(conn: Connection, plan_id: str) -> list[Attendance]:
    a = schema.attendance
    stmt = select(a).where(a.c.plan_id == plan_id).order_by(a.c.user_id)
    return [Attendance(**_row_dict(row)) for row in conn.execute(stmt)]


# --- messages ----------------------------------------------------------------------------------


def insert_message(
    conn: Connection,
    *,
    group_id: str,
    user_id: str,
    sent_at_ms: int,
    body: str,
    reply_to: str | None = None,
    tapback_target: str | None = None,
) -> Message:
    record = Message(
        id=_new_id(),
        group_id=group_id,
        user_id=user_id,
        sent_at_ms=sent_at_ms,
        body=body,
        reply_to=reply_to,
        tapback_target=tapback_target,
        char_count=len(body),
    )
    conn.execute(insert(schema.messages).values(**record.model_dump()))
    return record


def messages_for_group(conn: Connection, group_id: str) -> list[Message]:
    m = schema.messages
    stmt = select(m).where(m.c.group_id == group_id).order_by(m.c.sent_at_ms, m.c.id)
    return [Message(**_row_dict(r)) for r in conn.execute(stmt)]


# --- ratings (directed, §1 rule 2) -------------------------------------------------------------


def insert_rating(
    conn: Connection,
    plan_id: str,
    *,
    rater_id: str,
    ratee_id: str,
    would_see_again: bool,
    felt_connected_1_7: int | None = None,
) -> Rating:
    """Store the directed rating rater→ratee. The reverse direction is a separate row."""
    if rater_id == ratee_id:
        raise ValueError("ratings are directed between two different people")
    if felt_connected_1_7 is not None and not 1 <= felt_connected_1_7 <= 7:
        raise ValueError("felt_connected_1_7 must be within 1..7")
    record = Rating(
        plan_id=plan_id,
        rater_id=rater_id,
        ratee_id=ratee_id,
        would_see_again=would_see_again,
        felt_connected_1_7=felt_connected_1_7,
        submitted_at=_now(),
    )
    conn.execute(insert(schema.ratings).values(**record.model_dump()))
    return record


def ratings_for_plan(conn: Connection, plan_id: str) -> list[Rating]:
    r = schema.ratings
    stmt = select(r).where(r.c.plan_id == plan_id).order_by(r.c.rater_id, r.c.ratee_id)
    return [Rating(**_row_dict(row)) for row in conn.execute(stmt)]


# --- audit (§9) --------------------------------------------------------------------------------


def write_audit(
    conn: Connection,
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
    """Every bot decision writes one of these: why this group, why this removal, which versions."""
    record = AuditEntry(
        id=_new_id(),
        actor=actor,
        decision_type=decision_type,
        subject_user_id=subject_user_id,
        group_id=group_id,
        plan_id=plan_id,
        persona_version_id=persona_version_id,
        judge_version=judge_version,
        reason=reason,
        payload=dict(payload or {}),
        created_at=_now(),
    )
    values = record.model_dump()
    values["payload_json"] = values.pop("payload")
    conn.execute(insert(schema.audit_log).values(**values))
    return record


def _audit_from_row(data: dict[str, Any]) -> AuditEntry:
    data["payload"] = data.pop("payload_json")
    return AuditEntry(**data)


def get_audit(conn: Connection, audit_id: str) -> AuditEntry:
    a = schema.audit_log
    return _audit_from_row(_one(conn, select(a).where(a.c.id == audit_id), "audit_log"))


def audit_for_user(conn: Connection, user_id: str) -> list[AuditEntry]:
    a = schema.audit_log
    stmt = select(a).where(a.c.subject_user_id == user_id).order_by(a.c.created_at, a.c.id)
    return [_audit_from_row(_row_dict(r)) for r in conn.execute(stmt)]


# --- deposit ledger ----------------------------------------------------------------------------


def insert_deposit_credit(
    conn: Connection,
    *,
    user_id: str,
    amount_cents: int,
    kind: DepositKind,
    plan_id: str | None = None,
) -> DepositCredit:
    if kind not in DEPOSIT_KINDS:
        raise ValueError(f"kind must be one of {DEPOSIT_KINDS}, got {kind!r}")
    record = DepositCredit(
        id=_new_id(),
        user_id=user_id,
        plan_id=plan_id,
        amount_cents=amount_cents,
        kind=kind,
        created_at=_now(),
    )
    conn.execute(insert(schema.deposit_credits).values(**record.model_dump()))
    return record


def deposit_credits_for_user(conn: Connection, user_id: str) -> list[DepositCredit]:
    d = schema.deposit_credits
    stmt = select(d).where(d.c.user_id == user_id).order_by(d.c.created_at, d.c.id)
    return [DepositCredit(**_row_dict(r)) for r in conn.execute(stmt)]


def deposit_credits_query(
    conn: Connection, *, user_id: str | None = None, plan_id: str | None = None
) -> list[DepositCredit]:
    """Filtered read for ``LedgerStore.entries``: either filter may be omitted."""
    d = schema.deposit_credits
    stmt = select(d)
    if user_id is not None:
        stmt = stmt.where(d.c.user_id == user_id)
    if plan_id is not None:
        stmt = stmt.where(d.c.plan_id == plan_id)
    stmt = stmt.order_by(d.c.created_at, d.c.id)
    return [DepositCredit(**_row_dict(r)) for r in conn.execute(stmt)]


# --- bot state (0002; regulars.bot.store.BotStore over these) ----------------------------------


def bind_handle(
    conn: Connection, *, handle: str, user_id: str, display_name: str | None = None
) -> Handle:
    """Insert or update the handle -> user mapping. ``display_name`` defaults to the handle."""
    h = schema.handles
    record = Handle(
        handle=handle,
        user_id=user_id,
        display_name=display_name or handle,
        created_at=_now(),
    )
    values = record.model_dump()
    updated = conn.execute(update(h).where(h.c.handle == handle).values(**values)).rowcount
    if updated == 0:
        conn.execute(insert(h).values(**values))
    return record


def get_handle(conn: Connection, handle: str) -> Handle | None:
    h = schema.handles
    row = conn.execute(select(h).where(h.c.handle == handle)).first()
    return None if row is None else Handle(**_row_dict(row))


def handle_for_user(conn: Connection, user_id: str) -> Handle | None:
    h = schema.handles
    row = conn.execute(select(h).where(h.c.user_id == user_id).order_by(h.c.created_at)).first()
    return None if row is None else Handle(**_row_dict(row))


def save_group_chat(
    conn: Connection,
    *,
    group_id: str,
    chat_id: str,
    last_message_at_ms: int | None = None,
    last_sender: str | None = None,
) -> GroupChatRow:
    gc = schema.group_chats
    record = GroupChatRow(
        group_id=group_id,
        chat_id=chat_id,
        last_message_at_ms=last_message_at_ms,
        last_sender=last_sender,
    )
    values = record.model_dump()
    if conn.execute(update(gc).where(gc.c.group_id == group_id).values(**values)).rowcount == 0:
        conn.execute(insert(gc).values(**values))
    return record


def get_group_chat(conn: Connection, group_id: str) -> GroupChatRow | None:
    gc = schema.group_chats
    row = conn.execute(select(gc).where(gc.c.group_id == group_id)).first()
    return None if row is None else GroupChatRow(**_row_dict(row))


def group_chat_for_chat_id(conn: Connection, chat_id: str) -> GroupChatRow | None:
    gc = schema.group_chats
    row = conn.execute(select(gc).where(gc.c.chat_id == chat_id)).first()
    return None if row is None else GroupChatRow(**_row_dict(row))


def save_plan_record(conn: Connection, plan_id: str, plan_json: Mapping[str, Any]) -> PlanRecord:
    """Upsert the serialized ``regulars.bot.plan.Plan`` for ``plan_id`` (the plan is a record)."""
    pr = schema.plan_records
    record = PlanRecord(plan_id=plan_id, plan_json=dict(plan_json), updated_at=_now())
    values = record.model_dump()
    if conn.execute(update(pr).where(pr.c.plan_id == plan_id).values(**values)).rowcount == 0:
        conn.execute(insert(pr).values(**values))
    return record


def get_plan_record(conn: Connection, plan_id: str) -> PlanRecord | None:
    pr = schema.plan_records
    row = conn.execute(select(pr).where(pr.c.plan_id == plan_id)).first()
    return None if row is None else PlanRecord(**_row_dict(row))


def open_plan_records(conn: Connection) -> list[PlanRecord]:
    """Every stored plan whose JSON ``outcome`` is ``"open"``. Filtered in Python: the store's
    ``open_plans`` is called rarely (per ``tick``) and the plan count per tick is small."""
    pr = schema.plan_records
    rows = [PlanRecord(**_row_dict(r)) for r in conn.execute(select(pr))]
    return [r for r in rows if r.plan_json.get("outcome") == "open"]


def save_onboarding_state(
    conn: Connection, user_id: str, state_json: Mapping[str, Any]
) -> OnboardingStateRow:
    ob = schema.onboarding_state
    record = OnboardingStateRow(user_id=user_id, state_json=dict(state_json), updated_at=_now())
    values = record.model_dump()
    if conn.execute(update(ob).where(ob.c.user_id == user_id).values(**values)).rowcount == 0:
        conn.execute(insert(ob).values(**values))
    return record


def get_onboarding_state(conn: Connection, user_id: str) -> OnboardingStateRow | None:
    ob = schema.onboarding_state
    row = conn.execute(select(ob).where(ob.c.user_id == user_id)).first()
    return None if row is None else OnboardingStateRow(**_row_dict(row))


def save_morning_after_state(
    conn: Connection, *, user_id: str, plan_id: str, group_id: str, others: Sequence[str]
) -> MorningAfterStateRow:
    ma = schema.morning_after_state
    record = MorningAfterStateRow(
        user_id=user_id,
        plan_id=plan_id,
        group_id=group_id,
        others_json=list(others),
        updated_at=_now(),
    )
    values = record.model_dump()
    if conn.execute(update(ma).where(ma.c.user_id == user_id).values(**values)).rowcount == 0:
        conn.execute(insert(ma).values(**values))
    return record


def get_morning_after_state(conn: Connection, user_id: str) -> MorningAfterStateRow | None:
    ma = schema.morning_after_state
    row = conn.execute(select(ma).where(ma.c.user_id == user_id)).first()
    return None if row is None else MorningAfterStateRow(**_row_dict(row))


def clear_morning_after_state(conn: Connection, user_id: str) -> None:
    ma = schema.morning_after_state
    conn.execute(schema.morning_after_state.delete().where(ma.c.user_id == user_id))


def put_deposit_hold(
    conn: Connection,
    *,
    plan_id: str,
    user_id: str,
    intent_id: str | None,
    amount_cents: int,
    credit_cents: int = 0,
    state: HoldState = "held",
) -> DepositHoldRow:
    if state not in HOLD_STATES:
        raise ValueError(f"state must be one of {HOLD_STATES}, got {state!r}")
    dh = schema.deposit_holds
    record = DepositHoldRow(
        plan_id=plan_id,
        user_id=user_id,
        intent_id=intent_id,
        amount_cents=amount_cents,
        credit_cents=credit_cents,
        state=state,
        updated_at=_now(),
    )
    values = record.model_dump()
    key = and_(dh.c.plan_id == plan_id, dh.c.user_id == user_id)
    if conn.execute(update(dh).where(key).values(**values)).rowcount == 0:
        conn.execute(insert(dh).values(**values))
    return record


def deposit_holds_for_plan(conn: Connection, plan_id: str) -> list[DepositHoldRow]:
    dh = schema.deposit_holds
    stmt = select(dh).where(dh.c.plan_id == plan_id).order_by(dh.c.user_id)
    return [DepositHoldRow(**_row_dict(r)) for r in conn.execute(stmt)]


# --- webhook idempotency (0003; regulars.payments.idempotency.WebhookIdempotencyStore) ---------


def put_processed_webhook_event(
    conn: Connection,
    *,
    event_id: str,
    event_type: str,
    received_at: datetime,
    status: WebhookStatus,
    processed_at: datetime | None = None,
    error: str | None = None,
) -> ProcessedWebhookEventRow:
    """Insert or replace the row keyed by ``event_id``, mirroring ``put_deposit_hold``."""
    if status not in WEBHOOK_STATUSES:
        raise ValueError(f"status must be one of {WEBHOOK_STATUSES}, got {status!r}")
    pwe = schema.processed_webhook_events
    record = ProcessedWebhookEventRow(
        event_id=event_id,
        event_type=event_type,
        received_at=_utc(received_at) or received_at,
        processed_at=_utc(processed_at),
        status=status,
        error=error,
    )
    values = record.model_dump()
    key = pwe.c.event_id == event_id
    if conn.execute(update(pwe).where(key).values(**values)).rowcount == 0:
        conn.execute(insert(pwe).values(**values))
    return record


def get_processed_webhook_event(conn: Connection, event_id: str) -> ProcessedWebhookEventRow | None:
    pwe = schema.processed_webhook_events
    row = conn.execute(select(pwe).where(pwe.c.event_id == event_id)).first()
    return None if row is None else ProcessedWebhookEventRow(**_row_dict(row))
