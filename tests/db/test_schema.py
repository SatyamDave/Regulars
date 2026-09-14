from pathlib import Path

from sqlalchemy import Engine, inspect

from regulars.db.gen_ddl import render_migration_ddl, render_postgres_ddl
from regulars.db.schema import metadata

EXPECTED_0001_TABLES = {
    "users",
    "persona_versions",
    "sims",
    "sim_features",
    "predictions",
    "groups",
    "memberships",
    "plans",
    "rsvps",
    "attendance",
    "messages",
    "ratings",
    "audit_log",
    "deposit_credits",
}

EXPECTED_0002_TABLES = {
    "handles",
    "group_chats",
    "plan_records",
    "onboarding_state",
    "morning_after_state",
    "deposit_holds",
}

EXPECTED_TABLES = EXPECTED_0001_TABLES | EXPECTED_0002_TABLES

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "regulars/db/migrations"
MIGRATION = MIGRATIONS_DIR / "0001_initial.sql"
MIGRATION_0002 = MIGRATIONS_DIR / "0002_bot_state.sql"


def test_schema_creates_on_sqlite(engine: Engine) -> None:
    assert set(metadata.tables) == EXPECTED_TABLES
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_sqlite_has_persona_trigger(engine: Engine) -> None:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='trigger'").scalars()
        assert "persona_versions_immutable" in set(rows)


def test_checked_in_migration_matches_generator() -> None:
    assert MIGRATION.read_text() == render_postgres_ddl(), (
        "regenerate with: uv run python -m regulars.db.gen_ddl > "
        "regulars/db/migrations/0001_initial.sql"
    )


def test_checked_in_migration_0002_matches_generator() -> None:
    assert MIGRATION_0002.read_text() == render_migration_ddl("0002"), (
        "regenerate with: uv run python -m regulars.db.gen_ddl --migration 0002 > "
        "regulars/db/migrations/0002_bot_state.sql"
    )


def test_postgres_ddl_carries_the_three_rules() -> None:
    ddl = render_postgres_ddl()
    assert "CONSTRAINT uq_persona_versions_user_id_version UNIQUE (user_id, version)" in ddl
    assert "CONSTRAINT ck_ratings_directed CHECK (rater_id <> ratee_id)" in ddl
    assert "PRIMARY KEY (plan_id, rater_id, ratee_id)" in ddl
    assert "assembly_method VARCHAR(16) NOT NULL" in ddl
    assert "CHECK (assembly_method IN ('model', 'random'))" in ddl
    assert "PRIMARY KEY (group_id, user_id, joined_round)" in ddl
    assert "CREATE TRIGGER persona_versions_immutable BEFORE UPDATE ON persona_versions" in ddl
