"""``regulars.ops.seed`` against a real SQLite engine (no network, no LLM)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, select

from regulars.db import repo, schema
from regulars.db.engine import sqlite_memory_engine
from regulars.db.schema import ASSEMBLY_METHODS
from regulars.ops.seed import main, seed_database

N_USERS = 18
"""Three tables' worth: small enough to run fast, big enough for assemble_round to seat >=2
groups of GROUP_SIZE (6) so the post-dinner simulation has groups to act on."""


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


def test_seed_produces_expected_users_and_personas(engine: Engine) -> None:
    summary = seed_database(engine, n_users=N_USERS, zip_code="94110", seed=0)

    assert summary.n_users == N_USERS
    with engine.connect() as conn:
        users = repo.list_users(conn, zip_code="94110")
        assert len(users) == N_USERS
        for user in users:
            version = repo.latest_persona_version(conn, user.id)
            assert version is not None
            # §2/core.persona: friction_points/drains are load-bearing for the simulator.
            assert version.spec.has_negative_space()
            assert version.source == "voice_interview"


def test_seed_forms_groups_with_assembly_method_set(engine: Engine) -> None:
    summary = seed_database(engine, n_users=N_USERS, zip_code="94110", seed=1)

    assert summary.n_groups >= 2
    with engine.connect() as conn:
        rows = conn.execute(select(schema.groups.c.assembly_method)).scalars().all()
    assert len(rows) == summary.n_groups
    assert all(method in ASSEMBLY_METHODS for method in rows)


def test_seed_predictions_cover_prefiltered_pairs(engine: Engine) -> None:
    summary = seed_database(engine, n_users=N_USERS, zip_code="94110", seed=2)
    assert summary.n_predictions > 0


def test_seed_produces_a_prune_driven_removal_with_a_reason(engine: Engine) -> None:
    summary = seed_database(engine, n_users=N_USERS, zip_code="94110", seed=0)

    assert summary.n_rounds_simulated >= 1
    assert summary.n_removed >= 1
    with engine.connect() as conn:
        rows = conn.execute(
            select(schema.audit_log.c.reason, schema.audit_log.c.subject_user_id).where(
                schema.audit_log.c.decision_type == "member_removed"
            )
        ).all()
    assert rows, "expected at least one member_removed audit row"
    for reason, subject_user_id in rows:
        assert subject_user_id is not None
        assert reason.strip()


def test_main_refuses_a_nonempty_db_without_force(tmp_path: Path) -> None:
    db_path = tmp_path / "dev.db"
    db_url = f"sqlite:///{db_path}"

    exit_code = main(["--db-url", db_url, "--users", str(N_USERS), "--seed", "0"])
    assert exit_code == 0

    with_repeat = main(["--db-url", db_url, "--users", str(N_USERS), "--seed", "0"])
    assert with_repeat == 1

    # Refused: the second run must not have touched the data from the first.
    from regulars.db.engine import make_engine

    engine = make_engine(db_url)
    try:
        with engine.connect() as conn:
            users = repo.list_users(conn)
        assert len(users) == N_USERS
    finally:
        engine.dispose()


def test_main_seeds_with_force_on_a_nonempty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "dev.db"
    db_url = f"sqlite:///{db_path}"

    assert main(["--db-url", db_url, "--users", str(N_USERS), "--seed", "0"]) == 0
    assert main(["--db-url", db_url, "--users", "3", "--seed", "99", "--force"]) == 0

    from regulars.db.engine import make_engine

    engine = make_engine(db_url)
    try:
        with engine.connect() as conn:
            users = repo.list_users(conn)
        assert len(users) == N_USERS + 3
    finally:
        engine.dispose()
