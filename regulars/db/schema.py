"""Tables from technical-architecture.md §1, plus ``audit_log`` (§9) and ``deposit_credits``.

Portable across SQLite (tests) and Postgres (prod). Ids are UUID4 strings generated in Python so
that user ids line up with the ``str`` speaker ids used throughout ``regulars.core``. JSON columns
use ``sqlalchemy.JSON``. Timestamps are stored as UTC.

The three rules that matter more than the schema (§1):

1. ``persona_versions`` is append-only. Enforced by the unique key on (user_id, version), by the
   repository exposing no update path, and by a ``BEFORE UPDATE`` trigger on every dialect we run.
2. ``ratings`` are directed: (plan_id, rater_id, ratee_id) is the key and rater != ratee.
3. ``groups.assembly_method`` is NOT NULL and restricted to 'model' | 'random'.
"""

from __future__ import annotations

from typing import Literal, get_args

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.engine import Connection

from regulars.core.persona import PersonaSource

AssemblyMethod = Literal["model", "random"]
DepositKind = Literal["deposit_hold", "forfeit", "redistributed_credit", "applied"]
HoldState = Literal["held", "captured", "released"]

ASSEMBLY_METHODS: tuple[str, ...] = get_args(AssemblyMethod)
PERSONA_SOURCES: tuple[str, ...] = get_args(PersonaSource)
DEPOSIT_KINDS: tuple[str, ...] = get_args(DepositKind)
HOLD_STATES: tuple[str, ...] = get_args(HoldState)

# Tables added after the 0001 migration (see regulars/db/migrations/0002_bot_state.sql) are
# tagged with this marker so `gen_ddl.py`'s default output stays exactly the 0001 tables.
MIGRATION_0002 = "0002"

# Stable constraint names so the Postgres migration is diffable across regenerations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

ID = String(36)
TIMESTAMP = DateTime(timezone=True)


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


users = Table(
    "users",
    metadata,
    Column("id", ID, primary_key=True),
    Column("phone_hash", String(128), nullable=False, unique=True),
    Column("zip", String(16), nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Column("status", String(32), nullable=False, default="active"),
    Column("exploration_flag", Boolean, nullable=False, default=False),
)

persona_versions = Table(
    "persona_versions",
    metadata,
    Column("id", ID, primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), nullable=False),
    Column("version", Integer, nullable=False),
    Column("spec_json", JSON, nullable=False),
    Column("source", String(32), nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    UniqueConstraint("user_id", "version"),
    CheckConstraint(f"source IN ({_sql_in(PERSONA_SOURCES)})", name="source"),
    CheckConstraint("version >= 1", name="version_positive"),
)

sims = Table(
    "sims",
    metadata,
    Column("id", ID, primary_key=True),
    Column("user_a", ID, ForeignKey("users.id"), nullable=False),
    Column("user_b", ID, ForeignKey("users.id"), nullable=False),
    Column("persona_ver_a", ID, ForeignKey("persona_versions.id"), nullable=False),
    Column("persona_ver_b", ID, ForeignKey("persona_versions.id"), nullable=False),
    Column("scenario_id", String(64), nullable=False),
    Column("seed", Integer, nullable=False),
    Column("transcript_json", JSON, nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Index("ix_sims_pair", "user_a", "user_b"),
)

sim_features = Table(
    "sim_features",
    metadata,
    Column("sim_id", ID, ForeignKey("sims.id"), primary_key=True),
    Column("feature_name", String(64), primary_key=True),
    Column("value", Float, nullable=False),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", ID, primary_key=True),
    Column("user_a", ID, ForeignKey("users.id"), nullable=False),
    Column("user_b", ID, ForeignKey("users.id"), nullable=False),
    Column("judge_version", String(64), nullable=False),
    Column("y_hat", Float, nullable=False),
    Column("y_hat_calibrated", Float, nullable=True),
    Column("created_at", TIMESTAMP, nullable=False),
    Index("ix_predictions_pair", "user_a", "user_b"),
)

groups = Table(
    "groups",
    metadata,
    Column("id", ID, primary_key=True),
    Column("zip", String(16), nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Column("assembly_method", String(16), nullable=False),
    Column("status", String(32), nullable=False, default="active"),
    CheckConstraint(f"assembly_method IN ({_sql_in(ASSEMBLY_METHODS)})", name="assembly_method"),
)

memberships = Table(
    "memberships",
    metadata,
    Column("group_id", ID, ForeignKey("groups.id"), primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("joined_round", Integer, primary_key=True),
    Column("left_round", Integer, nullable=True),
    Column("exit_reason", String(64), nullable=True),
    CheckConstraint("left_round IS NULL OR left_round >= joined_round", name="left_after_joined"),
)

plans = Table(
    "plans",
    metadata,
    Column("id", ID, primary_key=True),
    Column("group_id", ID, ForeignKey("groups.id"), nullable=False),
    Column("proposed_at", TIMESTAMP, nullable=False),
    Column("locked_at", TIMESTAMP, nullable=True),
    Column("venue", String(255), nullable=True),
    Column("starts_at", TIMESTAMP, nullable=True),
    Index("ix_plans_group_id", "group_id"),
)

rsvps = Table(
    "rsvps",
    metadata,
    Column("plan_id", ID, ForeignKey("plans.id"), primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("state", String(32), nullable=False),
    Column("deposit_intent_id", String(255), nullable=True),
    Column("paid_at", TIMESTAMP, nullable=True),
)

attendance = Table(
    "attendance",
    metadata,
    Column("plan_id", ID, ForeignKey("plans.id"), primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("attended", Boolean, nullable=False),
    Column("source", String(32), nullable=False),
)

messages = Table(
    "messages",
    metadata,
    Column("id", ID, primary_key=True),
    Column("group_id", ID, ForeignKey("groups.id"), nullable=False),
    Column("user_id", ID, ForeignKey("users.id"), nullable=False),
    Column("sent_at_ms", BigInteger, nullable=False),
    Column("body", Text, nullable=False),
    Column("reply_to", ID, ForeignKey("messages.id"), nullable=True),
    Column("tapback_target", ID, ForeignKey("messages.id"), nullable=True),
    Column("char_count", Integer, nullable=False),
    Index("ix_messages_group_sent", "group_id", "sent_at_ms"),
)

ratings = Table(
    "ratings",
    metadata,
    Column("plan_id", ID, ForeignKey("plans.id"), primary_key=True),
    Column("rater_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("ratee_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("would_see_again", Boolean, nullable=False),
    Column("felt_connected_1_7", Integer, nullable=True),
    Column("submitted_at", TIMESTAMP, nullable=False),
    CheckConstraint("rater_id <> ratee_id", name="directed"),
    CheckConstraint(
        "felt_connected_1_7 IS NULL OR (felt_connected_1_7 BETWEEN 1 AND 7)",
        name="felt_connected_range",
    ),
)

audit_log = Table(
    "audit_log",
    metadata,
    Column("id", ID, primary_key=True),
    Column("actor", String(64), nullable=False),
    Column("decision_type", String(64), nullable=False),
    Column("subject_user_id", ID, ForeignKey("users.id"), nullable=True),
    Column("group_id", ID, ForeignKey("groups.id"), nullable=True),
    Column("plan_id", ID, ForeignKey("plans.id"), nullable=True),
    Column("persona_version_id", ID, ForeignKey("persona_versions.id"), nullable=True),
    Column("judge_version", String(64), nullable=True),
    Column("reason", Text, nullable=False),
    Column("payload_json", JSON, nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Index("ix_audit_log_subject_user_id", "subject_user_id"),
    Index("ix_audit_log_group_id", "group_id"),
    Index("ix_audit_log_created_at", "created_at"),
)

deposit_credits = Table(
    "deposit_credits",
    metadata,
    Column("id", ID, primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), nullable=False),
    Column("plan_id", ID, ForeignKey("plans.id"), nullable=True),
    Column("amount_cents", Integer, nullable=False),
    Column("kind", String(32), nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    CheckConstraint(f"kind IN ({_sql_in(DEPOSIT_KINDS)})", name="kind"),
    Index("ix_deposit_credits_user_id", "user_id"),
    Index("ix_deposit_credits_plan_id", "plan_id"),
)

# --- persona_versions immutability (§1 rule 1) -------------------------------------------------
# Belt and braces: the repository has no update path, and the database refuses one anyway.
# Run by ``metadata.create_all`` for the connection's dialect and printed by ``gen_ddl`` for
# Postgres. Dialects not listed here get no trigger (the API-level guarantee still holds).

PERSONA_IMMUTABLE_DDL: dict[str, tuple[str, ...]] = {
    "sqlite": (
        "CREATE TRIGGER persona_versions_immutable BEFORE UPDATE ON persona_versions\n"
        "BEGIN\n"
        "    SELECT RAISE(ABORT, 'persona_versions is append-only');\n"
        "END",
    ),
    "postgresql": (
        "CREATE OR REPLACE FUNCTION persona_versions_reject_update() RETURNS trigger AS $$\n"
        "BEGIN\n"
        "    RAISE EXCEPTION 'persona_versions is append-only';\n"
        "END;\n"
        "$$ LANGUAGE plpgsql",
        "CREATE TRIGGER persona_versions_immutable BEFORE UPDATE ON persona_versions\n"
        "FOR EACH ROW EXECUTE FUNCTION persona_versions_reject_update()",
    ),
}


@event.listens_for(persona_versions, "after_create")
def _create_persona_immutability_trigger(
    _target: Table, connection: Connection, **_kw: object
) -> None:
    for statement in PERSONA_IMMUTABLE_DDL.get(connection.dialect.name, ()):
        connection.execute(text(statement))


# --- 0002: bot + ledger state ------------------------------------------------------------------
# Everything the bot runtime (regulars.bot.store.BotStore) and the deposit ledger
# (regulars.payments.ledger.LedgerStore) need beyond the §1 tables above. Tagged
# ``info={"migration": MIGRATION_0002}`` so ``gen_ddl.py``'s default output (diffed against
# 0001_initial.sql in CI) is unaffected; see ``gen_ddl.py --migration 0002``.

handles = Table(
    "handles",
    metadata,
    Column("handle", String(255), primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), nullable=False),
    Column("display_name", String(255), nullable=False),
    Column("created_at", TIMESTAMP, nullable=False),
    Index("ix_handles_user_id", "user_id"),
    info={"migration": MIGRATION_0002},
)

group_chats = Table(
    "group_chats",
    metadata,
    Column("group_id", ID, ForeignKey("groups.id"), primary_key=True),
    Column("chat_id", String(255), nullable=False, unique=True),
    Column("last_message_at_ms", BigInteger, nullable=True),
    Column("last_sender", ID, ForeignKey("users.id"), nullable=True),
    info={"migration": MIGRATION_0002},
)

plan_records = Table(
    "plan_records",
    metadata,
    Column("plan_id", ID, ForeignKey("plans.id"), primary_key=True),
    Column("plan_json", JSON, nullable=False),
    Column("updated_at", TIMESTAMP, nullable=False),
    info={"migration": MIGRATION_0002},
)

onboarding_state = Table(
    "onboarding_state",
    metadata,
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("state_json", JSON, nullable=False),
    Column("updated_at", TIMESTAMP, nullable=False),
    info={"migration": MIGRATION_0002},
)

morning_after_state = Table(
    "morning_after_state",
    metadata,
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("plan_id", ID, ForeignKey("plans.id"), nullable=False),
    Column("group_id", ID, ForeignKey("groups.id"), nullable=False),
    Column("others_json", JSON, nullable=False),
    Column("updated_at", TIMESTAMP, nullable=False),
    info={"migration": MIGRATION_0002},
)

deposit_holds = Table(
    "deposit_holds",
    metadata,
    Column("plan_id", ID, ForeignKey("plans.id"), primary_key=True),
    Column("user_id", ID, ForeignKey("users.id"), primary_key=True),
    Column("intent_id", String(255), nullable=True),
    Column("amount_cents", Integer, nullable=False),
    Column("credit_cents", Integer, nullable=False, default=0),
    Column("state", String(16), nullable=False),
    Column("updated_at", TIMESTAMP, nullable=False),
    CheckConstraint(f"state IN ({_sql_in(HOLD_STATES)})", name="state"),
    info={"migration": MIGRATION_0002},
)
