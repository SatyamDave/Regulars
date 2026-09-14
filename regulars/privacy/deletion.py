"""Data deletion (§10 Privacy architecture; GDPR Art. 17, CCPA right-to-delete).

Two modes, chosen by the caller:

- **``anonymize`` (default).** Scrubs the user's own behavioral content in place but keeps every
  row and id, so nothing anyone else's data points at goes missing. This is the safe default
  because §1 rule 1 makes ``persona_versions`` append-only *for reproducibility*: another user's
  prediction, sim, or audit row may cite this user's ``persona_version_id``, and a hard delete of
  that row would silently corrupt someone else's ability to reproduce why a call was made. Kenny's
  Social Relations Model (§1 rule 2, technical-architecture.md §5) needs the *count and structure*
  of directed ratings intact to estimate actor/partner variance for everyone else in a completed
  round-robin — deleting this user's outgoing ratings entirely would bias other people's variance
  estimates, so they are anonymized (rater scrubbed to a tombstone id) rather than removed.
- **``hard``.** A genuine purge for when the law requires one (a formal deletion request past any
  retention exception). This necessarily breaks §1 rule 1's reproducibility guarantee for anything
  that referenced this user: predictions and sims involving them, and the audit rows that cite
  their subject or persona version, are cascade-deleted rather than left as dangling foreign keys.
  A single audit row documents that the purge happened, with no PII in it at all — not even the
  deleted user id — because the entire point of hard mode is that nothing should remain to link
  back to the person.

Both modes are irreversible in practice (anonymize does not restore the original spec/messages;
hard mode removes rows outright) and both leave a paper trail: anonymize leaves the normal audit
row via ``regulars.db.repo.write_audit`` describing the request, hard leaves the PII-free one
above.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import Connection, Engine, delete, or_, select, text, update

from regulars.core.persona import PersonaSpec, Provenance, Stated
from regulars.db import repo, schema

DeletionMode = Literal["hard", "anonymize"]

TOMBSTONE_MARKER = "[removed at user's request]"
"""What survives in a scrubbed ``persona_versions.spec_json`` and ``messages.body``."""

RATER_TOMBSTONE_PREFIX = "deleted-user:"
"""Anonymized ratings are re-keyed to ``deleted-user:<uuid>``. ``ratings.rater_id`` has a foreign
key into ``users``, so the tombstone id must itself be a real (placeholder) row — see
``_anonymize_ratings_given`` — not just an arbitrary string. One fresh tombstone id per deletion
call keeps it unlinkable to the real person while preserving the row count and directed structure
the Social Relations Model needs."""


def _tombstone_spec_json() -> dict[str, object]:
    """A structurally valid, content-free ``PersonaSpec``. Kept valid (not an arbitrary dict) so
    every existing reader — ``repo.get_persona_version``, ``legibility.render_plain_english``,
    the judge, the sim harness — can still parse the row instead of raising."""
    spec = PersonaSpec(
        stated=Stated(seeking=TOMBSTONE_MARKER),
        provenance=Provenance(sources=[], confidence_by_field={}),
    )
    return spec.model_dump(mode="json")


def _disable_persona_immutability(conn: Connection) -> None:
    """The append-only trigger (schema.py) blocks *every* UPDATE on ``persona_versions``,
    including this one. Anonymization is the one sanctioned exception to that guarantee — it is
    reachable only through this module, is itself audited, and never changes ``version`` or
    ``created_at``, only ``spec_json``."""
    dialect = conn.engine.dialect.name
    if dialect == "sqlite":
        conn.execute(text("DROP TRIGGER IF EXISTS persona_versions_immutable"))
    elif dialect == "postgresql":
        conn.execute(
            text("ALTER TABLE persona_versions DISABLE TRIGGER persona_versions_immutable")
        )
    # Other dialects never had the trigger created (see schema.PERSONA_IMMUTABLE_DDL); nothing
    # to disable.


def _reenable_persona_immutability(conn: Connection) -> None:
    dialect = conn.engine.dialect.name
    if dialect == "sqlite":
        for statement in schema.PERSONA_IMMUTABLE_DDL["sqlite"]:
            conn.execute(text(statement))
    elif dialect == "postgresql":
        conn.execute(text("ALTER TABLE persona_versions ENABLE TRIGGER persona_versions_immutable"))


class DeletionReport(BaseModel, frozen=True):
    """What actually happened, for the caller to show the user or log."""

    user_id: str
    mode: DeletionMode
    persona_versions_scrubbed: int = 0
    messages_scrubbed: int = 0
    ratings_given_anonymized: int = 0
    audit_rows_scrubbed: int = 0
    rows_deleted: dict[str, int] = Field(default_factory=dict)
    deletion_audit_id: str | None = None
    performed_at: datetime


def _anonymize_persona_versions(conn: Connection, user_id: str) -> int:
    pv = schema.persona_versions
    ids = [row.id for row in conn.execute(select(pv.c.id).where(pv.c.user_id == user_id))]
    if not ids:
        return 0
    _disable_persona_immutability(conn)
    try:
        conn.execute(
            update(pv).where(pv.c.user_id == user_id).values(spec_json=_tombstone_spec_json())
        )
    finally:
        _reenable_persona_immutability(conn)
    return len(ids)


def _anonymize_messages(conn: Connection, user_id: str, *, scrub_metadata: bool) -> int:
    m = schema.messages
    values: dict[str, object] = {"body": TOMBSTONE_MARKER}
    if scrub_metadata:
        values["reply_to"] = None
        values["tapback_target"] = None
    result = conn.execute(update(m).where(m.c.user_id == user_id).values(**values))
    return result.rowcount


def _anonymize_ratings_given(conn: Connection, user_id: str) -> int:
    """Re-key every outgoing rating to one fresh, unlinkable tombstone rater id.

    Deleting the rows instead would shrink the denominator the SRM decomposition uses for every
    ratee's *partner effect* and for the round's actor-variance estimate; re-keying keeps the
    count and the (rater != ratee) directed structure while destroying the link back to this
    person. ``ratings.rater_id`` has a foreign key into ``users``, so the tombstone id must be a
    real row, not an arbitrary string — a minimal placeholder user is inserted for it. One
    tombstone id per call (not per row) is enough: this user's own ratings were already unique on
    (plan_id, ratee_id), and a call only ever touches one user's outgoing ratings, so re-keying
    them all to the same fresh id cannot collide with itself or with another deletion's tombstone.
    """
    r = schema.ratings
    rows = list(conn.execute(select(r.c.plan_id, r.c.ratee_id).where(r.c.rater_id == user_id)))
    if not rows:
        return 0
    tombstone_id = f"{RATER_TOMBSTONE_PREFIX}{uuid.uuid4()}"
    repo.create_user(
        conn,
        phone_hash=f"tombstone:{uuid.uuid4()}",
        zip_code="00000",
        status="deleted",
        user_id=tombstone_id,
    )
    for plan_id, ratee_id in rows:
        conn.execute(
            update(r)
            .where(
                (r.c.plan_id == plan_id) & (r.c.rater_id == user_id) & (r.c.ratee_id == ratee_id)
            )
            .values(rater_id=tombstone_id)
        )
    return len(rows)


def _scrub_audit_subject(conn: Connection, user_id: str) -> int:
    """Audit rows are permanent (§9) even after anonymization — dispute resolution is exactly the
    case they exist for — but the free-text ``reason``/``payload_json`` a bot wrote may itself
    quote persona content. We keep the row and its linkage (``subject_user_id`` stays put; that is
    not behavioral content, just a foreign key) and only redact the free-text fields."""
    a = schema.audit_log
    result = conn.execute(
        update(a)
        .where(a.c.subject_user_id == user_id)
        .values(reason=TOMBSTONE_MARKER, payload_json={})
    )
    return result.rowcount


async def _anonymize(
    conn: Connection, user_id: str, *, scrub_message_metadata: bool
) -> dict[str, int]:
    persona_versions_scrubbed = _anonymize_persona_versions(conn, user_id)
    messages_scrubbed = _anonymize_messages(conn, user_id, scrub_metadata=scrub_message_metadata)
    ratings_given_anonymized = _anonymize_ratings_given(conn, user_id)
    audit_rows_scrubbed = _scrub_audit_subject(conn, user_id)
    conn.execute(update(schema.users).where(schema.users.c.id == user_id).values(status="deleted"))
    return {
        "persona_versions_scrubbed": persona_versions_scrubbed,
        "messages_scrubbed": messages_scrubbed,
        "ratings_given_anonymized": ratings_given_anonymized,
        "audit_rows_scrubbed": audit_rows_scrubbed,
    }


def _hard_delete(conn: Connection, user_id: str) -> dict[str, int]:
    """Cascade-delete everything that would otherwise dangle, then the user row itself.

    Order matters under foreign-key enforcement (children before parents). Rows belonging to
    *other* users are only ever edited to drop a reference (e.g. ``group_chats.last_sender`` is
    nulled, never deleted — the chat itself is not this user's to remove), never deleted outright.
    """
    counts: dict[str, int] = {}

    persona_ids = [
        row.id
        for row in conn.execute(
            select(schema.persona_versions.c.id).where(schema.persona_versions.c.user_id == user_id)
        )
    ]

    # audit_log rows that cite this user or one of their persona versions.
    a = schema.audit_log
    audit_conditions = [a.c.subject_user_id == user_id]
    if persona_ids:
        audit_conditions.append(a.c.persona_version_id.in_(persona_ids))
    counts["audit_log"] = conn.execute(delete(a).where(or_(*audit_conditions))).rowcount

    # sims (and their features) that used this user's personas, or paired this user directly.
    sims = schema.sims
    sim_conditions = [sims.c.user_a == user_id, sims.c.user_b == user_id]
    if persona_ids:
        sim_conditions.append(sims.c.persona_ver_a.in_(persona_ids))
        sim_conditions.append(sims.c.persona_ver_b.in_(persona_ids))
    sim_where = or_(*sim_conditions)
    sim_ids = [row.id for row in conn.execute(select(sims.c.id).where(sim_where))]
    if sim_ids:
        counts["sim_features"] = conn.execute(
            delete(schema.sim_features).where(schema.sim_features.c.sim_id.in_(sim_ids))
        ).rowcount
    counts["sims"] = conn.execute(delete(sims).where(sim_where)).rowcount

    p = schema.predictions
    counts["predictions"] = conn.execute(
        delete(p).where((p.c.user_a == user_id) | (p.c.user_b == user_id))
    ).rowcount

    r = schema.ratings
    counts["ratings"] = conn.execute(
        delete(r).where((r.c.rater_id == user_id) | (r.c.ratee_id == user_id))
    ).rowcount

    counts["attendance"] = conn.execute(
        delete(schema.attendance).where(schema.attendance.c.user_id == user_id)
    ).rowcount
    counts["rsvps"] = conn.execute(
        delete(schema.rsvps).where(schema.rsvps.c.user_id == user_id)
    ).rowcount

    # Messages: null out any *other* message's reply/tapback pointer into this user's messages
    # before deleting them, or the self-referencing foreign key blocks the delete.
    m = schema.messages
    own_message_ids = [row.id for row in conn.execute(select(m.c.id).where(m.c.user_id == user_id))]
    if own_message_ids:
        conn.execute(update(m).where(m.c.reply_to.in_(own_message_ids)).values(reply_to=None))
        conn.execute(
            update(m).where(m.c.tapback_target.in_(own_message_ids)).values(tapback_target=None)
        )
    counts["messages"] = conn.execute(delete(m).where(m.c.user_id == user_id)).rowcount

    counts["memberships"] = conn.execute(
        delete(schema.memberships).where(schema.memberships.c.user_id == user_id)
    ).rowcount
    counts["deposit_credits"] = conn.execute(
        delete(schema.deposit_credits).where(schema.deposit_credits.c.user_id == user_id)
    ).rowcount
    counts["deposit_holds"] = conn.execute(
        delete(schema.deposit_holds).where(schema.deposit_holds.c.user_id == user_id)
    ).rowcount
    counts["handles"] = conn.execute(
        delete(schema.handles).where(schema.handles.c.user_id == user_id)
    ).rowcount
    counts["onboarding_state"] = conn.execute(
        delete(schema.onboarding_state).where(schema.onboarding_state.c.user_id == user_id)
    ).rowcount
    counts["morning_after_state"] = conn.execute(
        delete(schema.morning_after_state).where(schema.morning_after_state.c.user_id == user_id)
    ).rowcount

    # group_chats.last_sender is informational, not this user's row to remove.
    gc = schema.group_chats
    conn.execute(update(gc).where(gc.c.last_sender == user_id).values(last_sender=None))

    if persona_ids:
        counts["persona_versions"] = conn.execute(
            delete(schema.persona_versions).where(schema.persona_versions.c.user_id == user_id)
        ).rowcount

    counts["users"] = conn.execute(
        delete(schema.users).where(schema.users.c.id == user_id)
    ).rowcount
    return {k: v for k, v in counts.items() if v}


def _write_pii_free_deletion_record(conn: Connection) -> str:
    """A permanent, content-free record that a hard purge happened. Deliberately carries no
    ``subject_user_id`` (that FK would dangle the moment the user row is gone anyway) and no
    other identifier tying it back to who was deleted — that is the whole point of hard mode."""
    entry = repo.write_audit(
        conn,
        actor="regulars.privacy.deletion",
        decision_type="hard_delete_purge",
        reason="A user's data was permanently purged at their request. No identifying "
        "information is retained in this record by design.",
        payload={"deletion_record_id": str(uuid.uuid4())},
    )
    return entry.id


async def delete_user_data(
    engine: Engine,
    user_id: str,
    *,
    mode: DeletionMode = "anonymize",
    scrub_message_metadata: bool = False,
) -> DeletionReport:
    """Erase or anonymize everything the system holds about ``user_id``.

    ``async`` for symmetry with the export side and the FastAPI route; the repository layer is
    synchronous SQLAlchemy Core, so this completes within one ``engine.begin()`` transaction.

    ``mode="anonymize"`` (default, see module docstring) is safe to run without legal review: it
    preserves every row, id, and foreign key, and only scrubs this user's own behavioral content.
    ``scrub_message_metadata`` additionally clears ``reply_to``/``tapback_target`` on their
    messages (default ``False`` — kept for other users' feature extraction, per the task's
    documented tradeoff).

    ``mode="hard"`` is a genuine purge: it cascades to predictions, sims, and audit rows that cite
    this user (see ``_hard_delete``), then deletes the ``users`` row itself. Use it only when a
    hard deletion is actually required (a legal obligation with no applicable retention
    exception) — it destroys §1's reproducibility guarantee for anyone who was ever paired with
    this user.

    Raises ``regulars.db.repo.NotFound`` if ``user_id`` does not exist.
    """
    with engine.begin() as conn:
        repo.get_user(conn, user_id)  # NotFound if the id is bad; nothing below should run.

        if mode == "anonymize":
            counts = await _anonymize(conn, user_id, scrub_message_metadata=scrub_message_metadata)
            repo.write_audit(
                conn,
                actor="regulars.privacy.deletion",
                decision_type="anonymize",
                subject_user_id=user_id,
                reason="User requested deletion; anonymized in place per §10 (see "
                "regulars.privacy.deletion for the reproducibility tradeoff).",
                payload=counts,
            )
            return DeletionReport(
                user_id=user_id,
                mode="anonymize",
                persona_versions_scrubbed=counts["persona_versions_scrubbed"],
                messages_scrubbed=counts["messages_scrubbed"],
                ratings_given_anonymized=counts["ratings_given_anonymized"],
                audit_rows_scrubbed=counts["audit_rows_scrubbed"],
                performed_at=datetime.now(UTC),
            )

        rows_deleted = _hard_delete(conn, user_id)
        deletion_audit_id = _write_pii_free_deletion_record(conn)
        return DeletionReport(
            user_id=user_id,
            mode="hard",
            rows_deleted=rows_deleted,
            deletion_audit_id=deletion_audit_id,
            performed_at=datetime.now(UTC),
        )
