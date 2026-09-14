from datetime import UTC, datetime, timedelta

from sqlalchemy import Connection

from regulars.core import PersonaSpec
from regulars.db import repo


def test_audit_row_round_trips_payload(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], group: repo.Group
) -> None:
    a = users[0]
    pv = repo.append_persona_version(conn, a.id, PersonaSpec(), "voice_interview")
    payload = {
        "candidates": ["b", "c"],
        "pair_scores": {"a->b": 0.71, "a->c": 0.42},
        "min_pair": 0.42,
        "nested": {"flags": [True, False, None], "n": 6},
    }
    written = repo.write_audit(
        conn,
        actor="bot",
        decision_type="group_assembly",
        reason="max-min optimizer picked this table",
        payload=payload,
        subject_user_id=a.id,
        group_id=group.id,
        persona_version_id=pv.id,
        judge_version="v1-linear-2026-09",
    )

    read = repo.get_audit(conn, written.id)
    assert read == written
    assert read.payload == payload
    assert read.persona_version_id == pv.id
    assert read.judge_version == "v1-linear-2026-09"
    assert repo.audit_for_user(conn, a.id) == [written]


def test_audit_payload_defaults_empty(conn: Connection) -> None:
    written = repo.write_audit(conn, actor="bot", decision_type="noop", reason="nothing to do")
    assert repo.get_audit(conn, written.id).payload == {}


def test_audit_for_group_is_chronological(conn: Connection, group: repo.Group) -> None:
    first = repo.write_audit(
        conn, actor="bot", decision_type="group_assembly", reason="r1", group_id=group.id
    )
    second = repo.write_audit(
        conn, actor="bot", decision_type="member_removed", reason="r2", group_id=group.id
    )
    repo.write_audit(conn, actor="bot", decision_type="noop", reason="unrelated")
    assert repo.audit_for_group(conn, group.id) == [first, second]


def test_audit_decision_counts(conn: Connection, group: repo.Group) -> None:
    repo.write_audit(
        conn, actor="bot", decision_type="group_assembly", reason="r1", group_id=group.id
    )
    repo.write_audit(
        conn, actor="bot", decision_type="member_removed", reason="r2", group_id=group.id
    )
    repo.write_audit(
        conn, actor="bot", decision_type="member_removed", reason="r3", group_id=group.id
    )
    assert repo.audit_decision_counts(conn) == {"group_assembly": 1, "member_removed": 2}


def test_audit_decision_counts_filters_by_zip_and_since(conn: Connection) -> None:
    near = repo.create_group(conn, zip_code="94110", assembly_method="model")
    far = repo.create_group(conn, zip_code="10001", assembly_method="model")
    repo.write_audit(
        conn, actor="bot", decision_type="group_assembly", reason="r1", group_id=near.id
    )
    repo.write_audit(
        conn, actor="bot", decision_type="group_assembly", reason="r2", group_id=far.id
    )
    assert repo.audit_decision_counts(conn, zip_code="94110") == {"group_assembly": 1}
    future = datetime.now(UTC) + timedelta(days=1)
    assert repo.audit_decision_counts(conn, since=future) == {}


def test_judge_version_counts_from_predictions_and_audit(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], group: repo.Group
) -> None:
    a, b, _c = users
    repo.insert_prediction(conn, user_a=a.id, user_b=b.id, judge_version="v1", y_hat=0.5)
    repo.write_audit(
        conn,
        actor="bot",
        decision_type="group_assembly",
        reason="r1",
        group_id=group.id,
        judge_version="v1",
    )
    repo.write_audit(
        conn,
        actor="bot",
        decision_type="group_assembly",
        reason="r2",
        group_id=group.id,
        judge_version="v2",
    )
    assert repo.judge_version_counts(conn) == {
        "v1": {"predictions": 1, "audit": 1},
        "v2": {"predictions": 0, "audit": 1},
    }
