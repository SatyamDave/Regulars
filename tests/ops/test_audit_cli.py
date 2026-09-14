import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from regulars.db import repo
from regulars.db.engine import make_engine, sqlite_memory_engine
from regulars.db.schema import metadata
from regulars.ops import audit_cli

T0 = datetime(2026, 9, 10, 18, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def scenario(engine: Engine) -> dict[str, str]:
    """A group assembled, a member removed with a reason, and a plan locked (§9 story)."""
    with engine.begin() as conn:
        priya = repo.create_user(conn, phone_hash="h-priya", zip_code="94110", user_id="priya")
        sam = repo.create_user(conn, phone_hash="h-sam", zip_code="94110", user_id="sam")
        lee = repo.create_user(conn, phone_hash="h-lee", zip_code="94110", user_id="lee")
        group = repo.create_group(conn, zip_code="94110", assembly_method="model")
        repo.add_membership(conn, group.id, priya.id, joined_round=1)
        repo.add_membership(conn, group.id, sam.id, joined_round=1)
        repo.add_membership(conn, group.id, lee.id, joined_round=1)
        repo.write_audit(
            conn,
            actor="bot",
            decision_type="group_assembly",
            reason="max-min optimizer picked this table",
            payload={"min_pair": 0.61},
            subject_user_id=priya.id,
            group_id=group.id,
            judge_version="v1-linear-2026-09",
        )
        repo.insert_prediction(
            conn, user_a=priya.id, user_b=sam.id, judge_version="v1-linear-2026-09", y_hat=0.7
        )
        repo.end_membership(conn, group.id, lee.id, left_round=2, exit_reason="removed")
        repo.write_audit(
            conn,
            actor="bot",
            decision_type="member_removed",
            reason="Two of five said they would not see you again.",
            subject_user_id=lee.id,
            group_id=group.id,
            judge_version="v1-linear-2026-09",
        )
        plan = repo.create_plan(conn, group.id)
        repo.lock_plan(conn, plan.id, venue="Bar Tartine", starts_at=T0 + timedelta(days=3))
        repo.write_audit(
            conn,
            actor="bot",
            decision_type="plan_locked",
            reason="quorum reached, venue confirmed",
            group_id=group.id,
            plan_id=plan.id,
        )
    return {"group_id": group.id, "plan_id": plan.id, "lee": lee.id, "priya": priya.id}


def test_user_returns_rows_newest_first(engine: Engine, scenario: dict[str, str]) -> None:
    entries = audit_cli.run_user(engine, scenario["lee"])
    assert [e.decision_type for e in entries] == ["member_removed"]
    assert entries[0].reason == "Two of five said they would not see you again."


def test_user_filters_by_decision_type_and_limit(
    engine: Engine, scenario: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    entries = audit_cli.run_user(engine, scenario["priya"], decision_type="group_assembly", limit=1)
    assert len(entries) == 1
    assert entries[0].decision_type == "group_assembly"
    out = capsys.readouterr().out
    assert "group_assembly" in out
    assert "max-min optimizer" in out


def test_user_json_output_is_valid_and_matches_row(
    engine: Engine, scenario: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    entries = audit_cli.run_user(engine, scenario["lee"], as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert len(parsed) == 1
    assert parsed[0]["reason"] == entries[0].reason
    assert parsed[0]["id"] == entries[0].id
    assert parsed[0]["decision_type"] == "member_removed"


def test_group_returns_full_story_chronologically(engine: Engine, scenario: dict[str, str]) -> None:
    entries = audit_cli.run_group(engine, scenario["group_id"])
    assert [e.decision_type for e in entries] == [
        "group_assembly",
        "member_removed",
        "plan_locked",
    ]
    assert all(e.group_id == scenario["group_id"] for e in entries)


def test_why_removed_returns_exact_reason(engine: Engine, scenario: dict[str, str]) -> None:
    reason = audit_cli.run_why_removed(engine, scenario["lee"], scenario["group_id"])
    assert reason == "Two of five said they would not see you again."


def test_why_removed_none_when_never_removed(engine: Engine, scenario: dict[str, str]) -> None:
    assert audit_cli.run_why_removed(engine, scenario["priya"], scenario["group_id"]) is None


def test_why_removed_scoped_to_group(engine: Engine, scenario: dict[str, str]) -> None:
    assert audit_cli.run_why_removed(engine, scenario["lee"], "some-other-group") is None


def test_stats_counts_by_decision_type(engine: Engine, scenario: dict[str, str]) -> None:
    counts = audit_cli.run_stats(engine)
    assert counts == {"group_assembly": 1, "member_removed": 1, "plan_locked": 1}


def test_stats_filters_by_zip(engine: Engine, scenario: dict[str, str]) -> None:
    counts = audit_cli.run_stats(engine, zip_code="94110")
    assert counts == {"group_assembly": 1, "member_removed": 1, "plan_locked": 1}
    assert audit_cli.run_stats(engine, zip_code="00000") == {}


def test_stats_json_output(
    engine: Engine, scenario: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    audit_cli.run_stats(engine, as_json=True)
    out = capsys.readouterr().out
    assert json.loads(out) == {"group_assembly": 1, "member_removed": 1, "plan_locked": 1}


def test_judge_versions_counts_predictions_and_audit(
    engine: Engine, scenario: dict[str, str]
) -> None:
    counts = audit_cli.run_judge_versions(engine)
    assert counts == {"v1-linear-2026-09": {"predictions": 1, "audit": 2}}


def test_main_dispatches_user_subcommand(tmp_path) -> None:
    # main() opens its own engine from --engine-url, so it needs a real file, not :memory:.
    db_path = tmp_path / "audit.db"
    url = f"sqlite+pysqlite:///{db_path}"
    seeding_engine = make_engine(url)
    metadata.create_all(seeding_engine)
    with seeding_engine.begin() as conn:
        user = repo.create_user(conn, phone_hash="h-u1", zip_code="94110", user_id="u1")
        repo.write_audit(
            conn,
            actor="bot",
            decision_type="noop",
            reason="nothing to do",
            subject_user_id=user.id,
        )
    seeding_engine.dispose()

    code = audit_cli.main(["user", "u1", "--engine-url", url, "--json"])
    assert code == 0


def test_user_since_filters_out_older_rows(engine: Engine, scenario: dict[str, str]) -> None:
    future = T0 + timedelta(days=365)
    entries = audit_cli.run_user(engine, scenario["lee"], since=future)
    assert entries == []


def test_render_entries_empty() -> None:
    assert audit_cli.render_entries([]) == "(no audit rows)"


def test_judge_versions_empty_and_stats_empty_render(engine: Engine) -> None:
    assert audit_cli.run_judge_versions(engine) == {}
    assert audit_cli.run_stats(engine) == {}


def test_main_dispatches_group_stats_and_judge_versions(tmp_path) -> None:
    db_path = tmp_path / "audit3.db"
    url = f"sqlite+pysqlite:///{db_path}"
    seeding_engine = make_engine(url)
    metadata.create_all(seeding_engine)
    with seeding_engine.begin() as conn:
        group = repo.create_group(conn, zip_code="94110", assembly_method="model")
        repo.write_audit(
            conn, actor="bot", decision_type="group_assembly", reason="r1", group_id=group.id
        )
        group_id = group.id
    seeding_engine.dispose()

    assert audit_cli.main(["group", group_id, "--engine-url", url]) == 0
    assert audit_cli.main(["stats", "--engine-url", url, "--json"]) == 0
    assert audit_cli.main(["judge-versions", "--engine-url", url]) == 0


def test_main_why_removed_exit_code(tmp_path) -> None:
    db_path = tmp_path / "audit2.db"
    url = f"sqlite+pysqlite:///{db_path}"
    seeding_engine = make_engine(url)
    metadata.create_all(seeding_engine)
    seeding_engine.dispose()

    code = audit_cli.main(["why-removed", "nobody", "nowhere", "--engine-url", url])
    assert code == 1
