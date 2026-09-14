from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import Connection, Engine

from regulars.core.persona import PersonaSpec, Social
from regulars.core.transcript import Transcript, Turn
from regulars.db import repo
from regulars.db.engine import sqlite_memory_engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


@dataclass
class Scenario:
    """Two real users (a, b) with a rich footprint each, so an export/deletion of ``a`` can be
    checked both for completeness and for leaving ``b`` untouched."""

    a: repo.User
    b: repo.User
    group: repo.Group
    plan: repo.Plan
    persona_a1: repo.PersonaVersion
    persona_a2: repo.PersonaVersion
    persona_b: repo.PersonaVersion
    message_a: repo.Message
    message_b: repo.Message
    sim_id: str
    prediction_id: str


def build_scenario(conn: Connection) -> Scenario:
    a = repo.create_user(conn, phone_hash="hash-a", zip_code="94110", user_id="user-a")
    b = repo.create_user(conn, phone_hash="hash-b", zip_code="94110", user_id="user-b")

    persona_a1 = repo.append_persona_version(
        conn, a.id, PersonaSpec(social=Social(friction_points=["flakiness"])), "voice_interview"
    )
    persona_a2 = repo.append_persona_version(
        conn, a.id, PersonaSpec(social=Social(drains=["small talk"])), "chat_behavior"
    )
    persona_b = repo.append_persona_version(conn, b.id, PersonaSpec(), "voice_interview")

    group = repo.create_group(conn, zip_code="94110", assembly_method="model")
    repo.add_membership(conn, group.id, a.id, joined_round=1)
    repo.add_membership(conn, group.id, b.id, joined_round=1)

    plan = repo.create_plan(conn, group.id, venue="Cafe X")

    message_a = repo.insert_message(
        conn, group_id=group.id, user_id=a.id, sent_at_ms=1000, body="hey I'm running late"
    )
    message_b = repo.insert_message(
        conn,
        group_id=group.id,
        user_id=b.id,
        sent_at_ms=2000,
        body="b's private thoughts nobody else should export",
        reply_to=message_a.id,
    )

    repo.insert_rating(conn, plan.id, rater_id=a.id, ratee_id=b.id, would_see_again=True)
    repo.insert_rating(
        conn, plan.id, rater_id=b.id, ratee_id=a.id, would_see_again=False, felt_connected_1_7=3
    )

    repo.insert_deposit_credit(conn, user_id=a.id, amount_cents=2000, kind="deposit_hold")

    repo.write_audit(
        conn,
        actor="bot",
        decision_type="group_open",
        subject_user_id=a.id,
        group_id=group.id,
        reason="paired a and b",
        payload={"note": "hello"},
    )

    transcript = Transcript(
        participants=[a.id, b.id],
        turns=[Turn(speaker_id=a.id, text="hi"), Turn(speaker_id=b.id, text="hey")],
        source="sim",
    )
    sim = repo.insert_sim(
        conn,
        user_a=a.id,
        user_b=b.id,
        persona_ver_a=persona_a2.id,
        persona_ver_b=persona_b.id,
        scenario_id="first_dinner",
        seed=1,
        transcript=transcript,
    )
    prediction = repo.insert_prediction(
        conn, user_a=a.id, user_b=b.id, judge_version="v1", y_hat=0.7
    )

    return Scenario(
        a=a,
        b=b,
        group=group,
        plan=plan,
        persona_a1=persona_a1,
        persona_a2=persona_a2,
        persona_b=persona_b,
        message_a=message_a,
        message_b=message_b,
        sim_id=sim.id,
        prediction_id=prediction.id,
    )


@pytest.fixture
def scenario(engine: Engine) -> Scenario:
    with engine.begin() as conn:
        return build_scenario(conn)
