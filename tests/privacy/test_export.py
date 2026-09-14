from __future__ import annotations

import json

import pytest
from sqlalchemy import Engine

from regulars.db.repo import NotFound
from regulars.privacy.export import DataExport, build_data_export
from tests.privacy.conftest import Scenario


async def test_export_contains_every_expected_section(engine: Engine, scenario: Scenario) -> None:
    export = await build_data_export(engine, scenario.a.id)

    assert export.user_id == scenario.a.id
    assert export.zip == "94110"

    versions = {pv.id: pv for pv in export.persona_versions}
    assert {scenario.persona_a1.id, scenario.persona_a2.id} == set(versions)
    assert versions[scenario.persona_a1.id].version == 1
    assert versions[scenario.persona_a2.id].version == 2
    assert versions[scenario.persona_a1.id].spec.social.friction_points == ["flakiness"]

    assert len(export.ratings_given) == 1
    assert export.ratings_given[0].ratee_id == scenario.b.id
    assert export.ratings_given[0].would_see_again is True

    assert len(export.ratings_received) == 1
    assert export.ratings_received[0].rater_id == scenario.b.id
    assert export.ratings_received[0].felt_connected_1_7 == 3

    assert [m.group_id for m in export.memberships] == [scenario.group.id]

    assert len(export.deposit_ledger) == 1
    assert export.deposit_ledger[0].amount_cents == 2000

    assert len(export.audit_log) == 1
    assert export.audit_log[0].decision_type == "group_open"


async def test_export_excludes_other_users_message_content(
    engine: Engine, scenario: Scenario
) -> None:
    export = await build_data_export(engine, scenario.a.id)

    bodies = [m.body for m in export.messages]
    assert bodies == ["hey I'm running late"]
    assert not any("private thoughts" in b for b in bodies)
    assert all(m.group_id != "" for m in export.messages)

    # b's own export, symmetrically, must not carry a's message content either.
    b_export = await build_data_export(engine, scenario.b.id)
    b_bodies = [m.body for m in b_export.messages]
    assert b_bodies == ["b's private thoughts nobody else should export"]
    assert "running late" not in " ".join(b_bodies)


async def test_export_round_trips_to_valid_json(engine: Engine, scenario: Scenario) -> None:
    export = await build_data_export(engine, scenario.a.id)
    raw = export.to_json()
    parsed = json.loads(raw)
    assert parsed["user_id"] == scenario.a.id
    restored = DataExport.model_validate(parsed)
    assert restored == export


async def test_export_unknown_user_raises_not_found(engine: Engine) -> None:
    with pytest.raises(NotFound):
        await build_data_export(engine, "does-not-exist")
