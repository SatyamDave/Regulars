import math

import pytest

from regulars.eval.records import Attendance, Membership
from regulars.eval.retention import round2_retention
from tests.eval.synth import make_retention


def test_retention_diff_and_ci_on_15pp_gap() -> None:
    s = make_retention(n_groups_per_arm=50, p_model=0.75, p_random=0.60, seed=0)
    res = round2_retention(s.memberships, s.attendance, reps=2000, seed=0)
    assert res.n_model == 300 and res.n_random == 300
    assert res.n_groups_model == 50 and res.n_groups_random == 50
    assert res.diff_pp == pytest.approx(15.0, abs=6.0)
    lo, hi = res.ci95
    assert lo < res.diff_pp < hi
    assert lo > 0.0  # the planted lift is detectable at this n
    assert hi - lo < 25.0
    assert res.pooled_rate == pytest.approx((res.model_rate + res.random_rate) / 2)


def test_dead_group_counts_members_as_not_retained() -> None:
    memberships = [
        Membership(group_id="g", user_id=f"u{i}", joined_round=1, assembly_method="random")
        for i in range(6)
    ] + [
        Membership(group_id="h", user_id=f"v{i}", joined_round=1, assembly_method="model")
        for i in range(6)
    ]
    attendance = [
        Attendance(plan_id="h-2", group_id="h", user_id=f"v{i}", round_index=2, attended=True)
        for i in range(3)
    ]
    res = round2_retention(memberships, attendance, reps=10)
    assert res.random_rate == 0.0
    assert res.model_rate == 0.5
    assert res.diff_pp == 50.0
    assert math.isnan(res.ci95[0])  # fewer than 2 groups per arm: no bootstrap


def test_round_boundaries_and_joiners() -> None:
    memberships = [
        Membership(group_id="g", user_id="a", joined_round=1, assembly_method="random"),
        Membership(
            group_id="g", user_id="b", joined_round=1, left_round=1, assembly_method="random"
        ),
        Membership(group_id="g", user_id="c", joined_round=2, assembly_method="random"),
    ]
    attendance = [
        Attendance(plan_id="p", group_id="g", user_id="a", round_index=2, attended=True),
        Attendance(plan_id="p", group_id="g", user_id="c", round_index=2, attended=True),
    ]
    res = round2_retention(memberships, attendance, reps=10)
    # b left before round 1 happened and c joined in round 2: only a is a round-1 member
    assert res.n_random == 1
    assert res.random_rate == 1.0
    assert res.n_model == 0 and math.isnan(res.model_rate)


def test_mixed_assembly_method_in_one_group_rejected() -> None:
    memberships = [
        Membership(group_id="g", user_id="a", joined_round=1, assembly_method="random"),
        Membership(group_id="g", user_id="b", joined_round=1, assembly_method="model"),
    ]
    with pytest.raises(ValueError, match="mixed assembly_method"):
        round2_retention(memberships, [], reps=10)
