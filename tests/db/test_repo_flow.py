from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection
from sqlalchemy.exc import IntegrityError

from regulars.core import FEATURE_NAMES, PersonaSpec, Transcript, Turn
from regulars.core.features import FOLLOW_UP_RATE, LSM_MEAN
from regulars.db import repo


def test_sim_features_and_prediction(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    a, b, _ = users
    pa = repo.append_persona_version(conn, a.id, PersonaSpec(), "voice_interview")
    pb = repo.append_persona_version(conn, b.id, PersonaSpec(), "voice_interview")
    transcript = Transcript(
        participants=[a.id, b.id],
        turns=[
            Turn(speaker_id=a.id, text="hi"),
            Turn(speaker_id=b.id, text="hey, how's it going?"),
        ],
        source="sim",
    )
    sim = repo.insert_sim(
        conn,
        user_a=a.id,
        user_b=b.id,
        persona_ver_a=pa.id,
        persona_ver_b=pb.id,
        scenario_id="first_dinner",
        seed=7,
        transcript=transcript,
    )
    assert sim.transcript == transcript

    repo.insert_sim_features(conn, sim.id, {FOLLOW_UP_RATE: 0.5, LSM_MEAN: 0.8})
    assert repo.sim_features_for(conn, sim.id) == {FOLLOW_UP_RATE: 0.5, LSM_MEAN: 0.8}

    with pytest.raises(ValueError, match="unknown feature"):
        repo.insert_sim_features(conn, sim.id, {"vibes": 1.0})
    assert "vibes" not in FEATURE_NAMES

    pred = repo.insert_prediction(conn, user_a=a.id, user_b=b.id, judge_version="v1", y_hat=0.6)
    assert pred.y_hat_calibrated is None

    with pytest.raises(IntegrityError):  # persona versions must exist
        repo.insert_sim(
            conn,
            user_a=a.id,
            user_b=b.id,
            persona_ver_a="missing",
            persona_ver_b=pb.id,
            scenario_id="x",
            seed=1,
            transcript=transcript,
        )


def test_plan_rsvp_attendance(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], group: repo.Group
) -> None:
    a, b, _ = users
    plan = repo.create_plan(conn, group.id)
    assert plan.locked_at is None

    starts = datetime(2026, 10, 1, 19, 0, tzinfo=UTC)
    locked = repo.lock_plan(conn, plan.id, venue="Nopa", starts_at=starts)
    assert locked.locked_at is not None
    assert locked.venue == "Nopa"
    assert locked.starts_at == starts
    with pytest.raises(ValueError, match="already locked"):
        repo.lock_plan(conn, plan.id, venue="Elsewhere", starts_at=starts)
    with pytest.raises(repo.NotFound):
        repo.lock_plan(conn, "nope", venue="Elsewhere", starts_at=starts)

    first = repo.upsert_rsvp(conn, plan.id, a.id, state="yes", deposit_intent_id="pi_1")
    assert first.paid_at is None
    paid = repo.upsert_rsvp(
        conn, plan.id, a.id, state="yes", deposit_intent_id="pi_1", paid_at=datetime.now(UTC)
    )
    assert paid.paid_at is not None
    repo.upsert_rsvp(conn, plan.id, b.id, state="no")

    assert repo.record_attendance(conn, plan.id, a.id, attended=True, source="bot").attended
    corrected = repo.record_attendance(conn, plan.id, a.id, attended=False, source="human")
    assert (corrected.attended, corrected.source) == (False, "human")


def test_messages_ordered_and_threaded(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], group: repo.Group
) -> None:
    a, b, _ = users
    m1 = repo.insert_message(conn, group_id=group.id, user_id=a.id, sent_at_ms=2000, body="hello")
    m2 = repo.insert_message(
        conn, group_id=group.id, user_id=b.id, sent_at_ms=1000, body="early", reply_to=None
    )
    tap = repo.insert_message(
        conn, group_id=group.id, user_id=b.id, sent_at_ms=3000, body="❤️", tapback_target=m1.id
    )
    got = repo.messages_for_group(conn, group.id)
    assert [m.id for m in got] == [m2.id, m1.id, tap.id]
    assert got[1].char_count == 5
    assert got[2].tapback_target == m1.id


def test_deposit_ledger(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a = users[0]
    repo.insert_deposit_credit(
        conn, user_id=a.id, plan_id=plan.id, amount_cents=2000, kind="deposit_hold"
    )
    repo.insert_deposit_credit(
        conn, user_id=a.id, plan_id=plan.id, amount_cents=-2000, kind="forfeit"
    )
    repo.insert_deposit_credit(conn, user_id=a.id, amount_cents=500, kind="redistributed_credit")
    with pytest.raises(ValueError, match="kind"):
        repo.insert_deposit_credit(conn, user_id=a.id, amount_cents=1, kind="tip")  # type: ignore[arg-type]
    rows = repo.deposit_credits_for_user(conn, a.id)
    assert [r.kind for r in rows] == ["deposit_hold", "forfeit", "redistributed_credit"]
    assert sum(r.amount_cents for r in rows) == 500


def test_user_round_trip(conn: Connection) -> None:
    u = repo.create_user(conn, phone_hash="abc", zip_code="10001", exploration_flag=True)
    assert repo.get_user(conn, u.id) == u
    with pytest.raises(repo.NotFound):
        repo.get_user(conn, "missing")
    with pytest.raises(IntegrityError):  # phone_hash unique
        repo.create_user(conn, phone_hash="abc", zip_code="10002")
