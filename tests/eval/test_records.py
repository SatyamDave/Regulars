from pathlib import Path

import pytest
from pydantic import ValidationError

from regulars.eval.records import LabeledDyad, Membership, Rsvp, dump_jsonl, load_dyads


def _dyad(**kw: object) -> LabeledDyad:
    base: dict[str, object] = {
        "group_id": "g",
        "plan_id": "p",
        "rater": "a",
        "ratee": "b",
        "y_hat": 0.3,
        "would_see_again": True,
        "assembly_method": "random",
        "round_index": 1,
    }
    base.update(kw)
    return LabeledDyad.model_validate(base)


def test_outcome_prefers_felt_connected() -> None:
    assert _dyad().outcome == 1.0
    assert _dyad(would_see_again=False).outcome == 0.0
    assert _dyad(felt_connected_1_7=6).outcome == 6.0


def test_validation_is_strict() -> None:
    with pytest.raises(ValidationError):
        _dyad(assembly_method="greedy")
    with pytest.raises(ValidationError):
        _dyad(felt_connected_1_7=9)
    with pytest.raises(ValidationError):
        _dyad(extra_field=1)
    with pytest.raises(ValidationError):
        _dyad().y_hat = 1.0  # type: ignore[misc]


def test_membership_rounds_and_rsvp_commitment() -> None:
    m = Membership(group_id="g", user_id="u", joined_round=2, left_round=4, assembly_method="model")
    assert not m.member_in_round(1)
    assert m.member_in_round(2) and m.member_in_round(3)
    assert not m.member_in_round(4)
    assert not m.active
    assert Rsvp(plan_id="p", user_id="u", state="YES").committed
    assert Rsvp(plan_id="p", user_id="u", state="maybe", paid=True).committed
    assert not Rsvp(plan_id="p", user_id="u", state="declined").committed


def test_jsonl_round_trip_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    dyads = [_dyad(rater="a", ratee="b"), _dyad(rater="b", ratee="a", felt_connected_1_7=2)]
    dump_jsonl(path, dyads)
    path.write_text(path.read_text() + "\n\n")
    assert load_dyads(path) == dyads
