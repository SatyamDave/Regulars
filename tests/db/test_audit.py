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
