"""End-to-end gate (product-plan §9 Weekend 1): a stranger texts the number and ends up in a
locked, paid plan — through the REAL FastAPI routes (``regulars.wiring.build_system``) and the
REAL SQLite-backed stores (``SqlBotStore``, ``SqlLedgerStore``), never the in-memory test doubles.

Everything is offline: ``FakeTransport`` stands in for BlueBubbles, ``FrozenClock`` for time, and
a ``FakeStripeClient`` behind a real ``StripeDeposits`` (see ``regulars.payments.fake_stripe``)
stands in for Stripe's webhook signature check — so ``POST /webhooks/stripe`` is the real route,
just with a verifier that trusts the body instead of checking a real signature.

One long scenario, in the four parts the task lays out; each part's assertions build on state
the previous part left in SQLite.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from regulars.app import Settings
from regulars.bot.clock import FrozenClock
from regulars.bot.events import OnboardingEvent
from regulars.bot.onboarding import OnboardingState
from regulars.bot.plan import Slot
from regulars.bot.store import Member
from regulars.bot.transport import FakeTransport, dm_chat_id
from regulars.core.llm import FakeLLM
from regulars.db import repo
from regulars.db.engine import sqlite_memory_engine
from regulars.persona.extract import extract_persona
from regulars.persona.interview import InterviewAnswer as PersonaInterviewAnswer
from regulars.pipeline import assemble_and_open, post_dinner
from regulars.wiring import System, build_system, fake_stripe_event

T0 = datetime(2026, 9, 14, 17, 0, tzinfo=UTC)

PERSONA_REPLY = json.dumps(
    {
        "conversational": {
            "opener_style": "asks what you've been up to",
            "avg_turn_length": "medium",
            "humor_register": "dry",
            "question_tendency": "high",
            "interrupt_tendency": "waits for a pause",
            "silence_tolerance": "comfortable with a lull",
        },
        "content": {
            "unprompted_topics": ["hiking"],
            "avoided_topics": [],
            "escalates_on": ["politics"],
            "deflects_on": [],
        },
        "social": {
            "energy": "initiator",
            "disclosure_speed": "fast",
            "friction_points": ["talks over the quietest person at the table"],
            "drains": ["small talk that never goes anywhere"],
        },
        "stated": {"exclusions": [], "seeking": "a regular friend group"},
        "confidence": {},
    }
)


def _bb_message(
    chat_id: str, handle: str, text: str, at: datetime, *, audio: bool = False
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "guid": f"m-{handle}-{int(at.timestamp() * 1000)}",
        "text": text,
        "dateCreated": int(at.timestamp() * 1000),
        "isFromMe": False,
        "handle": {"address": handle},
        "chats": [{"guid": chat_id}],
    }
    if audio:
        data["attachments"] = [{"guid": f"att-{data['guid']}", "mimeType": "audio/m4a"}]
    return {"type": "new-message", "data": data}


def _post(client: TestClient, payload: dict[str, Any]) -> None:
    r = client.post("/webhooks/bluebubbles", content=json.dumps(payload))
    assert r.status_code == 200


def _make_system() -> tuple[System, TestClient, FrozenClock]:
    clock = FrozenClock(T0)
    transport = FakeTransport()
    engine = sqlite_memory_engine()
    system = build_system(Settings(), engine=engine, transport=transport, clock=clock)
    return system, TestClient(system.app), clock


async def test_stranger_to_locked_paid_plan() -> None:
    system, client, clock = _make_system()
    engine = system.engine

    # --- (a) onboarding: welcome -> three audio answers -> persona version -------------------
    handle = "+15550001000"
    chat_id = dm_chat_id(handle)

    _post(client, _bb_message(chat_id, handle, "hi", clock.current))
    user_id = system.store.user_for_handle(handle)
    assert user_id is not None
    onboarding = system.store.get_onboarding(user_id)
    assert onboarding is not None and onboarding.stage == "prompt_1"

    answers = [
        "Went hiking with a few friends, cooked a big dinner after, no phones.",
        "Stopped hanging out with a guy who always talked over the quietest person there.",
        "A regular friend group I don't have to plan three weeks in advance to see.",
    ]
    for text in answers:
        clock.advance(timedelta(minutes=1))
        _post(client, _bb_message(chat_id, handle, text, clock.current, audio=True))

    onboarding = system.store.get_onboarding(user_id)
    assert onboarding is not None and onboarding.done

    onboarding_events = [e for e in system.runtime.events if isinstance(e, OnboardingEvent)]
    assert len(onboarding_events) == 1
    event = onboarding_events[0]
    assert event.user_id == user_id
    assert len(event.answers) == 3

    persona_answers = [
        PersonaInterviewAnswer(prompt_index=a.prompt_index, text=a.text, typed=a.typed)
        for a in event.answers
    ]
    fake_llm = FakeLLM(replies=[PERSONA_REPLY])
    spec = await extract_persona(fake_llm, persona_answers)
    assert spec.has_negative_space()
    with engine.begin() as conn:
        repo.append_persona_version(conn, user_id, spec, "voice_interview")
        assert repo.latest_persona_version(conn, user_id) is not None

    # --- (b) six users -> assemble_and_open -> GroupOpener -----------------------------------
    names = ["Priya", "Dev", "Anika", "Rohan", "Meera", "Sam"]
    members: list[Member] = []
    for i, name in enumerate(names, start=1):
        h = f"+155500020{i:02d}"
        uid = system.store.ensure_user(h)
        system.store.save_onboarding(OnboardingState(user_id=uid, stage="done"))
        members.append(Member(uid, h, name))
    zip_code = "00000"  # regulars.db.bot_store.PLACEHOLDER_ZIP: every ensure_user lands here

    with engine.begin() as conn:
        ids = [m.user_id for m in members]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                repo.insert_prediction(conn, user_a=a, user_b=b, judge_version="v1-test", y_hat=1.0)

    opened = assemble_and_open(
        engine, zip_code, random.Random(7), exploration_rate=0.0, group_size=6
    )
    assert len(opened) == 1
    group = opened[0]
    assert group.assembly_method == "model"
    with engine.begin() as conn:
        group_row = repo.get_group(conn, group.group_id)
        assert group_row.assembly_method == "model"

    hooks = {m.user_id: f"{m.name} always finds the best hole-in-the-wall place" for m in members}
    slots = [
        Slot(id="s1", venue="Rooh", starts_at=clock.current + timedelta(days=3)),
        Slot(id="s2", venue="Besharam", starts_at=clock.current + timedelta(days=4)),
    ]
    deadline = clock.current + timedelta(hours=24)
    plan = await system.runtime.flows.opener.open(
        group.group_id, members, hooks, slots, deadline=deadline
    )
    group_chat = system.store.get_group(group.group_id)
    assert group_chat is not None
    transport = system.runtime.tools.transport
    assert isinstance(transport, FakeTransport)
    bodies = transport.sent_to(group_chat.chat_id)
    assert len(bodies) == 2  # intro, then the proposal
    assert "Rooh" in bodies[1]

    # --- (c) vote -> tick -> deposits -> Stripe payments -> one drop -> lock -----------------
    for m in members[:5]:
        clock.advance(timedelta(minutes=1))
        _post(client, _bb_message(group_chat.chat_id, m.handle, "1", clock.current))

    clock.advance(timedelta(hours=24, minutes=1))
    await system.runtime.tick(clock.current)

    plan = system.store.get_plan(plan.id)
    assert plan is not None
    assert plan.locked_slot_id == "s1"
    assert plan.deposit_requested_at is not None

    for m in members[:5]:
        intent_id = plan.deposit_intents[m.user_id]
        body = fake_stripe_event(
            "payment_intent.amount_capturable_updated",
            plan_id=plan.id,
            user_id=m.user_id,
            intent_id=intent_id,
        )
        r = client.post(
            "/webhooks/stripe", content=body, headers={"Stripe-Signature": "test-does-not-check"}
        )
        assert r.status_code == 200

    clock.advance(timedelta(hours=24, minutes=1))
    backfills = await system.runtime.tick(clock.current)
    assert len(backfills) == 1
    assert backfills[0].dropped == (members[5].user_id,)

    plan = system.store.get_plan(plan.id)
    assert plan is not None
    assert plan.outcome == "locked"
    assert plan.locked_at is not None

    with engine.begin() as conn:
        plan_row = repo.get_plan(conn, plan.id)
        assert plan_row.locked_at is not None
        rsvp_states = {r.user_id: r.state for r in repo.rsvps_for_plan(conn, plan.id)}
        for m in members[:5]:
            assert rsvp_states[m.user_id] == "paid"
        assert rsvp_states[members[5].user_id] == "dropped"

        dropped_audit = repo.audit_for_user(conn, members[5].user_id)
        assert any(a.decision_type == "member_removed" for a in dropped_audit)
        assert any(a.decision_type == "collect_deposit" for a in dropped_audit)

    # --- (d) morning after: directed ratings -> post_dinner removes the low-rated member ------
    attendees = members[:5]  # members[5] was already dropped for non-payment
    with engine.begin() as conn:
        for m in attendees:
            repo.record_attendance(conn, plan.id, m.user_id, attended=True, source="self_report")

    await system.runtime.flows.morning_after.start(plan, [m.user_id for m in attendees])

    group_names = system.store.get_group(group.group_id)
    assert group_names is not None
    display = group_names.names()

    low_rated = attendees[4]  # everyone but one attendee leaves them out of "again"
    positive_rater = attendees[3]

    def again_line(rater: Member, exclude: set[str]) -> str:
        others = [m.user_id for m in attendees if m.user_id != rater.user_id]
        keep = [uid for uid in others if uid not in exclude]
        return "again: " + ", ".join(display[uid] for uid in keep)

    raters_and_exclusions = [
        (attendees[0], {low_rated.user_id}),
        (attendees[1], {low_rated.user_id}),
        (attendees[2], {low_rated.user_id}),
        (positive_rater, set()),
    ]
    for rater, exclude in raters_and_exclusions:
        clock.advance(timedelta(minutes=1))
        reply = again_line(rater, exclude)
        _post(client, _bb_message(dm_chat_id(rater.handle), rater.handle, reply, clock.current))

    with engine.begin() as conn:
        all_ratings = repo.ratings_for_plan(conn, plan.id)
        ratings_about_low_rated = [r for r in all_ratings if r.ratee_id == low_rated.user_id]
        assert len(ratings_about_low_rated) == 4
        approvals = sum(1 for r in ratings_about_low_rated if r.would_see_again)
        assert approvals == 1  # only positive_rater
        # directed, both directions present: ratings about other attendees too, not just the
        # one about to be removed (§1 rule 2: a->b is a different row from b->a).
        assert any(r.ratee_id != low_rated.user_id for r in all_ratings)

    update = post_dinner(engine, plan.id, queue=[], min_keep=3)
    assert low_rated.user_id in {r.user_id for r in update.removed}
    removal_reason = next(r.reason for r in update.removed if r.user_id == low_rated.user_id)
    assert "fewer than half" in removal_reason

    with engine.begin() as conn:
        active = {
            m.user_id for m in repo.memberships_for_group(conn, group.group_id, active_only=True)
        }
        assert low_rated.user_id not in active

    # She never answered her own morning-after prompt; clear it so "why was I removed" is read
    # as the standing removal-reason command rather than an unparsable morning-after reply.
    system.store.clear_morning_after(low_rated.user_id)
    why_removed = _bb_message(
        dm_chat_id(low_rated.handle), low_rated.handle, "why was I removed", clock.current
    )
    _post(client, why_removed)
    dms = transport.dms_to(low_rated.handle)
    assert any(removal_reason in body for body in dms)
