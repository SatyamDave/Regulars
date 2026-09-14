import math
import random

import pytest

from regulars.assembly.exploration import (
    GroupPlan,
    assemble_round,
    choose_method,
    leftover_users,
    random_groups,
)
from regulars.assembly.optimizer import PairScores, group_score
from regulars.assembly.prefilter import Candidate, Pair, pair_key, plausible_pairs


def test_choose_method_rate_is_about_30_percent() -> None:
    rng = random.Random(0)
    draws = [choose_method(rng) for _ in range(10_000)]
    assert set(draws) <= {"model", "random"}
    share = draws.count("random") / len(draws)
    assert share == pytest.approx(0.30, abs=0.02)


def test_choose_method_validates_rate() -> None:
    with pytest.raises(ValueError):
        choose_method(random.Random(0), exploration_rate=1.5)


def two_zip_users() -> tuple[list[str], frozenset[Pair]]:
    cands = [Candidate(f"x{i}", "1", frozenset({"fri"})) for i in range(6)] + [
        Candidate(f"y{i}", "2", frozenset({"fri"})) for i in range(6)
    ]
    return [c.user_id for c in cands], plausible_pairs(cands)


def test_random_groups_respect_plausibility() -> None:
    users, plausible = two_zip_users()
    for seed in range(20):
        rg = random_groups(users, 6, plausible, random.Random(seed))
        # The only valid partition has probability 2/C(12,6) per uniform shuffle, so the greedy
        # fallback (flagged relaxed=True) may be used; plausibility must hold either way.
        assert len(rg.groups) == 2
        assert rg.leftovers == ()
        for g in rg.groups:
            for i, a in enumerate(g):
                for b in g[i + 1 :]:
                    assert pair_key(a, b) in plausible
        assert all(r.payload["assembly_method"] == "random" for r in rg.audit)
        assert len(rg.audit) == len(rg.groups)


def test_random_groups_are_actually_random() -> None:
    users = [f"u{i}" for i in range(12)]
    plausible = frozenset(pair_key(a, b) for i, a in enumerate(users) for b in users[i + 1 :])
    results = [random_groups(users, 6, plausible, random.Random(s)) for s in range(30)]
    assert all(not r.relaxed and r.tries == 1 for r in results)
    seen = {frozenset(frozenset(g) for g in r.groups) for r in results}
    assert len(seen) > 1


def test_random_groups_fallback_never_seats_implausible_pair() -> None:
    # Only a-b and c-d plausible among five users; uniform shuffle of 2 groups of 2 almost never
    # lands on it, so with few tries the greedy fallback kicks in. It must still respect
    # plausibility and never seat e.
    users = ["a", "b", "c", "d", "e"]
    plausible = frozenset({("a", "b"), ("c", "d")})
    rg = random_groups(users, 2, plausible, random.Random(1), max_tries=1)
    for g in rg.groups:
        assert pair_key(*g) in plausible
    assert "e" in rg.leftovers
    assert rg.tries == 1
    if rg.relaxed:
        assert all(r.payload["relaxed"] for r in rg.audit)


def test_random_groups_too_few_users() -> None:
    rg = random_groups(["a", "b"], 6, frozenset({("a", "b")}), random.Random(0))
    assert rg.groups == () and rg.leftovers == ("a", "b")


def all_plans_valid(plans: list[GroupPlan], plausible: frozenset[Pair]) -> None:
    seated: list[str] = []
    for p in plans:
        assert p.assembly_method in {"model", "random"}
        assert len(p.members) == 6
        seated.extend(p.members)
        for i, a in enumerate(p.members):
            for b in p.members[i + 1 :]:
                assert pair_key(a, b) in plausible
        kinds = [r.decision_type for r in p.audit]
        assert kinds == ["method_chosen", "group_assembled"]
        assert all(r.group_members == p.members for r in p.audit)
        assert all(r.payload["assembly_method"] == p.assembly_method for r in p.audit)
    assert len(seated) == len(set(seated))


def test_assemble_round_sets_method_per_group_and_respects_plausibility() -> None:
    users, plausible = two_zip_users()
    rng = random.Random(4)
    scores = PairScores({k: rng.random() for k in plausible})
    methods: set[str] = set()
    for seed in range(40):
        plans = assemble_round(users, scores, plausible, [], random.Random(seed), 0.5)
        all_plans_valid(plans, plausible)
        assert leftover_users(users, plans) == ()
        for p in plans:
            methods.add(p.assembly_method)
            assert p.min_score == pytest.approx(group_score(p.members, scores))
            assert p.min_score > -math.inf
    assert methods == {"model", "random"}


def test_assemble_round_model_arm_never_uses_implausible_score() -> None:
    # a high score on an implausible (cross-zip) pair must not pull that pair together
    users, plausible = two_zip_users()
    scores = PairScores({k: 0.5 for k in plausible})
    scores.set("x0", "y0", 10.0)
    plans = assemble_round(users, scores, plausible, [], random.Random(0), exploration_rate=0.0)
    assert all(p.assembly_method == "model" for p in plans)
    all_plans_valid(plans, plausible)


def test_assemble_round_random_arm_only_when_rate_is_one() -> None:
    users, plausible = two_zip_users()
    scores = PairScores({k: 0.5 for k in plausible})
    plans = assemble_round(users, scores, plausible, [], random.Random(0), exploration_rate=1.0)
    assert len(plans) == 2
    assert all(p.assembly_method == "random" for p in plans)
    all_plans_valid(plans, plausible)


def test_assemble_round_continuity_applies_to_model_arm() -> None:
    users = ["a", "b", "c", "d"]
    plausible = frozenset(pair_key(a, b) for i, a in enumerate(users) for b in users[i + 1 :])
    scores = PairScores({k: 0.5 for k in plausible})
    plans = assemble_round(
        users,
        scores,
        plausible,
        previous_groups=[{"a", "c"}, {"b", "d"}],
        rng=random.Random(0),
        exploration_rate=0.0,
        group_size=2,
    )
    assert {frozenset(p.members) for p in plans} == {frozenset("ac"), frozenset("bd")}
