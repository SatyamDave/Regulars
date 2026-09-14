from __future__ import annotations

import pytest
from sqlalchemy import Engine, select

from regulars.db import repo, schema
from regulars.db.repo import NotFound
from regulars.persona import legibility
from regulars.privacy.deletion import RATER_TOMBSTONE_PREFIX, TOMBSTONE_MARKER, delete_user_data
from tests.privacy.conftest import Scenario


async def test_anonymize_scrubs_target_and_preserves_fk_integrity(
    engine: Engine, scenario: Scenario
) -> None:
    report = await delete_user_data(engine, scenario.a.id, mode="anonymize")

    assert report.mode == "anonymize"
    assert report.persona_versions_scrubbed == 2
    assert report.messages_scrubbed == 1
    assert report.ratings_given_anonymized == 1

    with engine.begin() as conn:
        # persona spec content is gone, but both versions and their ids/versions survive.
        for version_id in (scenario.persona_a1.id, scenario.persona_a2.id):
            pv = repo.get_persona_version(conn, version_id)
            assert pv.spec.stated.seeking == TOMBSTONE_MARKER
            assert pv.spec.social.friction_points == []
            assert pv.spec.social.drains == []
        assert repo.get_persona_version(conn, scenario.persona_a1.id).version == 1
        assert repo.get_persona_version(conn, scenario.persona_a2.id).version == 2

        # legibility rendering still works against a scrubbed spec (no crash on read-back).
        latest = repo.latest_persona_version(conn, scenario.a.id)
        assert latest is not None
        rendered = legibility.render_plain_english(latest.spec)
        assert TOMBSTONE_MARKER in rendered

        # a's message body is scrubbed; the row (and its id) still exists.
        messages = repo.messages_by_user(conn, scenario.a.id)
        assert len(messages) == 1
        assert messages[0].body == TOMBSTONE_MARKER
        assert messages[0].id == scenario.message_a.id

        # b's message, referencing a's (now scrubbed) message via reply_to, is untouched.
        b_messages = repo.messages_by_user(conn, scenario.b.id)
        assert b_messages[0].body == "b's private thoughts nobody else should export"
        assert b_messages[0].reply_to == scenario.message_a.id

        # a's outgoing rating is re-keyed to an unlinkable tombstone rater id.
        ratings_given = repo.ratings_given_by(conn, scenario.a.id)
        assert ratings_given == []
        r = schema.ratings
        rows = list(
            conn.execute(
                select(r).where((r.c.plan_id == scenario.plan.id) & (r.c.ratee_id == scenario.b.id))
            )
        )
        assert len(rows) == 1
        assert rows[0].rater_id.startswith(RATER_TOMBSTONE_PREFIX)
        assert rows[0].rater_id != scenario.a.id

        # b's rating *of* a (a rating received) is untouched — the SRM directionality is intact.
        ratings_received = repo.ratings_received_by(conn, scenario.a.id)
        assert len(ratings_received) == 1
        assert ratings_received[0].rater_id == scenario.b.id

        # the user row itself is marked deleted, not removed.
        a_user = repo.get_user(conn, scenario.a.id)
        assert a_user.status == "deleted"

        # b's data is completely untouched.
        b_user = repo.get_user(conn, scenario.b.id)
        assert b_user.status == "active"
        b_persona = repo.get_persona_version(conn, scenario.persona_b.id)
        assert b_persona.spec.stated.seeking != TOMBSTONE_MARKER

        # sims/predictions pairing a and b still resolve — nothing orphaned.
        assert (
            conn.execute(
                select(schema.sims.c.id).where(schema.sims.c.id == scenario.sim_id)
            ).first()
            is not None
        )
        assert (
            conn.execute(
                select(schema.predictions.c.id).where(
                    schema.predictions.c.id == scenario.prediction_id
                )
            ).first()
            is not None
        )


async def test_anonymize_can_also_clear_message_structure_when_asked(
    engine: Engine, scenario: Scenario
) -> None:
    await delete_user_data(engine, scenario.a.id, mode="anonymize", scrub_message_metadata=True)
    with engine.begin() as conn:
        # b's message pointed at a's; a's own row's reply_to/tapback_target are what get cleared.
        a_messages = repo.messages_by_user(conn, scenario.a.id)
        assert a_messages[0].reply_to is None
        assert a_messages[0].tapback_target is None


async def test_hard_delete_removes_user_and_cascades(engine: Engine, scenario: Scenario) -> None:
    report = await delete_user_data(engine, scenario.a.id, mode="hard")

    assert report.mode == "hard"
    assert report.rows_deleted["users"] == 1
    assert report.deletion_audit_id is not None

    with engine.begin() as conn:
        with pytest.raises(NotFound):
            repo.get_user(conn, scenario.a.id)

        assert repo.persona_versions_for_user(conn, scenario.a.id) == []
        assert (
            conn.execute(
                select(schema.sims.c.id).where(schema.sims.c.id == scenario.sim_id)
            ).first()
            is None
        )
        assert (
            conn.execute(
                select(schema.predictions.c.id).where(
                    schema.predictions.c.id == scenario.prediction_id
                )
            ).first()
            is None
        )
        assert repo.ratings_given_by(conn, scenario.a.id) == []
        assert repo.memberships_for_user(conn, scenario.a.id) == []

        # b's message that replied to a's (now-deleted) message has its pointer nulled, not
        # itself deleted.
        b_messages = repo.messages_by_user(conn, scenario.b.id)
        assert len(b_messages) == 1
        assert b_messages[0].reply_to is None
        assert b_messages[0].body == "b's private thoughts nobody else should export"

        # b's own data (persona, membership, rating received) is completely intact.
        b_user = repo.get_user(conn, scenario.b.id)
        assert b_user.status == "active"
        assert repo.persona_versions_for_user(conn, scenario.b.id) != []
        assert repo.memberships_for_user(conn, scenario.b.id) != []

        # the one permanent deletion record carries no PII: no subject_user_id, generic reason.
        assert report.deletion_audit_id is not None
        entry = repo.get_audit(conn, report.deletion_audit_id)
        assert entry.subject_user_id is None
        assert scenario.a.id not in entry.reason
        assert scenario.a.id not in str(entry.payload)


async def test_hard_delete_unknown_user_raises_not_found(engine: Engine) -> None:
    with pytest.raises(NotFound):
        await delete_user_data(engine, "does-not-exist", mode="hard")
