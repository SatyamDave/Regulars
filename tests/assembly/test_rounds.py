import math

import pytest

from regulars.assembly.optimizer import PairScores
from regulars.assembly.rounds import (
    AttendanceRecord,
    DirectedRating,
    prune_and_backfill,
)

GROUP = ["a", "b", "c", "d", "e", "f"]


def attended(*users: str, paid: bool = True) -> list[AttendanceRecord]:
    return [AttendanceRecord(u, attended=True, paid=paid) for u in users]


def rate(ratee: str, verdicts: dict[str, bool]) -> list[DirectedRating]:
    return [DirectedRating(rater, ratee, v) for rater, v in verdicts.items()]


def full_scores(users: list[str], value: float = 0.5) -> PairScores:
    return PairScores({(a, b): value for i, a in enumerate(users) for b in users[i + 1 :]})


def test_no_show_after_paying_is_removed_with_reason() -> None:
    att = attended("a", "b", "c", "d", "e") + [AttendanceRecord("f", attended=False, paid=True)]
    upd = prune_and_backfill(GROUP, [], att, [], full_scores(GROUP))
    assert upd.keep == ("a", "b", "c", "d", "e")
    assert [u for u, _ in upd.removed] == ["f"]
    user_id, reason = upd.removed[0]
    assert user_id == "f"
    assert "deposit" in reason and "didn't come" in reason
    assert len(upd.audit) == 1
    rec = upd.audit[0]
    assert rec.decision_type == "member_removed"
    assert rec.subject_user_id == "f"
    assert rec.reason == reason
    assert rec.group_members == tuple(GROUP)
    assert rec.payload["rule"] == "no_show"


def test_unpaid_absentee_is_not_a_no_show() -> None:
    att = attended("a", "b", "c", "d", "e") + [AttendanceRecord("f", attended=False, paid=False)]
    upd = prune_and_backfill(GROUP, [], att, [], full_scores(GROUP))
    assert upd.removed == []
    assert upd.keep == tuple(GROUP)


def test_low_rated_member_removed_with_reason() -> None:
    ratings = rate("f", {"a": False, "b": False, "c": True, "d": False})
    upd = prune_and_backfill(GROUP, ratings, attended(*GROUP), [], full_scores(GROUP))
    assert [u for u, _ in upd.removed] == ["f"]
    reason = upd.removed[0].reason
    assert "1 of the 4 people" in reason and "fewer than half" in reason
    assert upd.audit[0].payload == {
        "rule": "low_rating",
        "raters": 4,
        "approvals": 1,
        "approval_share": 0.25,
        "min_raters": 3,
        "threshold": 0.5,
    }


def test_exactly_half_approval_is_kept() -> None:
    ratings = rate("f", {"a": False, "b": True, "c": False, "d": True})
    upd = prune_and_backfill(GROUP, ratings, attended(*GROUP), [], full_scores(GROUP))
    assert upd.removed == []


def test_fewer_than_three_raters_is_kept() -> None:
    ratings = rate("f", {"a": False, "b": False})
    upd = prune_and_backfill(GROUP, ratings, attended(*GROUP), [], full_scores(GROUP))
    assert upd.removed == []
    assert upd.keep == tuple(GROUP)
    assert upd.audit == []


def test_only_other_attendees_ratings_count() -> None:
    ratings = (
        rate("f", {"a": False, "b": False})
        + [DirectedRating("f", "f", False)]  # self-rating ignored
        + [DirectedRating("z", "f", False)]  # non-member ignored
        + [DirectedRating("e", "f", False)]  # e did not attend: ignored
    )
    att = attended("a", "b", "c", "d", "f") + [AttendanceRecord("e", attended=False)]
    upd = prune_and_backfill(GROUP, ratings, att, [], full_scores(GROUP))
    assert upd.removed == []


def test_ratings_are_directed() -> None:
    # f rates everyone badly; nobody rates f badly. f stays, nobody else is touched either
    # (each of them has only one negative rater).
    ratings = [DirectedRating("f", u, False) for u in GROUP if u != "f"]
    upd = prune_and_backfill(GROUP, ratings, attended(*GROUP), [], full_scores(GROUP))
    assert upd.removed == []


def test_min_keep_spares_removals_that_would_shrink_group_too_far() -> None:
    att = attended("a", "b", "c") + [
        AttendanceRecord(u, attended=False, paid=True) for u in ("d", "e", "f")
    ]
    ratings = rate("c", {"a": False, "b": False}) + rate("a", {"b": False, "c": False})
    upd = prune_and_backfill(GROUP, ratings, att, [], full_scores(GROUP), min_keep=4)
    assert [u for u, _ in upd.removed] == ["d", "e"]
    assert upd.keep == ("a", "b", "c", "f")
    assert len(upd.notes) == 1 and "f" in upd.notes[0]


def test_backfill_maximizes_new_group_min() -> None:
    group = ["a", "b", "c", "d"]
    att = attended("a", "b", "c") + [AttendanceRecord("d", attended=False, paid=True)]
    # q1 is first in the queue but fits worse than q2; q3 fits best but is unscored vs c.
    scores = full_scores(group, 0.9)
    for u, v in (("q1", 0.3), ("q2", 0.8)):
        for m in ("a", "b", "c"):
            scores.set(u, m, v)
    scores.set("q3", "a", 1.0)
    scores.set("q3", "b", 1.0)
    upd = prune_and_backfill(group, [], att, ["q1", "q3", "q2"], scores)
    assert [u for u, _ in upd.removed] == ["d"]
    assert upd.added == ("q2",)
    assert upd.members == ("a", "b", "c", "q2")
    assert upd.min_score == pytest.approx(0.8)
    added = [r for r in upd.audit if r.decision_type == "member_added"]
    assert len(added) == 1
    assert added[0].subject_user_id == "q2"
    assert added[0].group_members == ("a", "b", "c", "q2")
    assert added[0].payload["resulting_min"] == pytest.approx(0.8)
    assert added[0].payload["queue_position"] == 2
    assert added[0].payload["seat_of"] == ["d"]
    assert "open seat" in added[0].reason


def test_backfill_fills_multiple_seats_greedily_and_skips_unscored() -> None:
    group = ["a", "b", "c", "d"]
    att = attended("a", "b") + [AttendanceRecord(u, attended=False, paid=True) for u in "cd"]
    scores = PairScores({("a", "b"): 0.9})
    for u, v in (("q1", 0.5), ("q2", 0.7)):
        scores.set(u, "a", v)
        scores.set(u, "b", v)
    scores.set("q1", "q2", 0.6)
    # q4 is scored against a and b but not against q1/q2, so it can only be the first pick;
    # q2 beats it there, and afterwards q4 is unscored against q2 and is skipped.
    scores.set("q4", "a", 0.65)
    scores.set("q4", "b", 0.65)
    upd = prune_and_backfill(group, [], att, ["q4", "q1", "q2"], scores, min_keep=2)
    assert [u for u, _ in upd.removed] == ["c", "d"]
    assert upd.added == ("q2", "q1")
    assert upd.min_score == pytest.approx(0.5)
    assert upd.members == ("a", "b", "q1", "q2")


def test_backfill_stops_when_nobody_fits() -> None:
    att = attended("a", "b") + [AttendanceRecord("c", attended=False, paid=True)]
    upd = prune_and_backfill(
        ["a", "b", "c"], [], att, ["z"], PairScores({("a", "b"): 0.5}), min_keep=2
    )
    assert upd.added == ()
    assert upd.keep == ("a", "b")
    assert any("backfill stopped" in n for n in upd.notes)
    assert upd.min_score == 0.5


def test_removed_members_are_not_backfilled_from_queue() -> None:
    att = attended("a", "b") + [AttendanceRecord("c", attended=False, paid=True)]
    upd = prune_and_backfill(["a", "b", "c"], [], att, ["c"], full_scores(["a", "b", "c"]))
    assert upd.added == ()


def test_group_size_override_and_min_score_for_tiny_groups() -> None:
    upd = prune_and_backfill(["a"], [], [], [], PairScores(), group_size=1)
    assert upd.min_score == math.inf
    with pytest.raises(ValueError):
        prune_and_backfill(["a", "a"], [], [], [], PairScores())


def test_every_decision_has_an_audit_record() -> None:
    group = ["a", "b", "c", "d", "e", "f"]
    att = attended("a", "b", "c", "d", "e") + [AttendanceRecord("f", attended=False, paid=True)]
    ratings = rate("e", {"a": False, "b": False, "c": False})
    scores = full_scores([*group, "q1", "q2"], 0.5)
    upd = prune_and_backfill(group, ratings, att, ["q1", "q2"], scores)
    assert [u for u, _ in upd.removed] == ["f", "e"]  # no-show first, then low rating
    assert upd.added == ("q1", "q2")
    kinds = [(r.decision_type, r.subject_user_id) for r in upd.audit]
    assert kinds == [
        ("member_removed", "f"),
        ("member_removed", "e"),
        ("member_added", "q1"),
        ("member_added", "q2"),
    ]
    for r in upd.audit:
        assert r.reason.strip()
        assert r.group_members
