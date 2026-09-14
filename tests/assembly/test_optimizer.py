import math
import random

import pytest

from regulars.assembly.optimizer import (
    Continuity,
    PairScores,
    assemble_groups,
    group_score,
)


def complete_scores(users: list[str], value: float) -> dict[tuple[str, str], float]:
    return {(a, b): value for i, a in enumerate(users) for b in users[i + 1 :]}


def planted_two_cliques(seed: int = 1) -> tuple[list[str], list[str], PairScores, float]:
    rng = random.Random(seed)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    raw: dict[tuple[str, str], float] = {}
    for clique in (a, b):
        for i, x in enumerate(clique):
            for y in clique[i + 1 :]:
                raw[(x, y)] = rng.uniform(0.7, 0.9)
    for x in a:
        for y in b:
            raw[(x, y)] = rng.uniform(0.0, 0.3)
    planted_min = min(v for (x, y), v in raw.items() if x[0] == y[0])
    return a, b, PairScores(raw), planted_min


def test_pair_scores_symmetric_and_missing_is_neg_inf() -> None:
    s = PairScores({("b", "a"): 0.4})
    assert s.get("a", "b") == 0.4 == s.get("b", "a")
    assert s.get("a", "z") == -math.inf
    assert s.has("a", "b") and not s.has("a", "z")
    with pytest.raises(ValueError, match="conflicting"):
        PairScores({("a", "b"): 0.1, ("b", "a"): 0.2})


def test_group_score_is_min_not_mean() -> None:
    s = PairScores(
        {
            ("a", "b"): 0.9,
            ("a", "c"): 0.9,
            ("a", "d"): 0.9,
            ("b", "c"): 0.9,
            ("b", "d"): 0.9,
            ("c", "d"): 0.1,
        }
    )
    assert group_score(["a", "b", "c", "d"], s) == 0.1
    assert group_score(["a"], s) == math.inf
    assert group_score(["a", "z"], s) == -math.inf


def test_recovers_planted_two_cliques() -> None:
    a, b, scores, planted_min = planted_two_cliques()
    result = assemble_groups([*a, *b], scores, group_size=6, seed=3)
    assert {frozenset(g) for g in result.groups} == {frozenset(a), frozenset(b)}
    assert result.global_min == pytest.approx(planted_min)
    assert min(result.per_group_min) == pytest.approx(planted_min)
    assert result.leftovers == ()
    assert len(result.audit) == 2
    assert all(r.decision_type == "group_assembled" for r in result.audit)
    assert all(r.payload["assembly_method"] == "model" for r in result.audit)


def test_optimizer_prefers_higher_min_over_higher_mean() -> None:
    # {ab, cd}: mean 0.6, min 0.2.  {ac, bd}: mean 0.6, min 0.6.  Greedy seeds on ab (best
    # pair) and would settle on {ab, cd}; local search must swap to {ac, bd}.
    s = PairScores(
        {
            ("a", "b"): 1.0,
            ("c", "d"): 0.2,
            ("a", "c"): 0.6,
            ("b", "d"): 0.6,
            ("a", "d"): 0.5,
            ("b", "c"): 0.5,
        }
    )
    result = assemble_groups(["a", "b", "c", "d"], s, group_size=2)
    assert {frozenset(g) for g in result.groups} == {frozenset("ac"), frozenset("bd")}
    assert result.global_min == 0.6
    no_search = assemble_groups(["a", "b", "c", "d"], s, group_size=2, iterations=0)
    assert no_search.global_min == 0.2


def test_local_search_uses_leftovers() -> None:
    # e is a leftover but pairs far better with a than b does.
    s = PairScores(
        {("a", "b"): 0.3, ("a", "e"): 0.9, ("b", "e"): 0.1, ("c", "d"): 0.8, ("c", "e"): 0.0}
    )
    result = assemble_groups(["a", "b", "c", "d", "e"], s, group_size=2)
    assert frozenset("ae") in {frozenset(g) for g in result.groups}
    assert result.leftovers == ("b",)
    assert result.global_min == 0.8


def test_continuity_keeps_prior_group_when_tied() -> None:
    users = ["a", "b", "c", "d"]
    s = PairScores(complete_scores(users, 0.5))
    prior = [{"a", "c"}, {"b", "d"}]
    with_bonus = assemble_groups(users, s, group_size=2, continuity=Continuity(prior, 0.01))
    assert {frozenset(g) for g in with_bonus.groups} == {frozenset("ac"), frozenset("bd")}
    # reported minima are raw scores, without the bonus
    assert with_bonus.global_min == 0.5
    assert with_bonus.per_group_min == (0.5, 0.5)
    assert with_bonus.audit[0].payload["continuity_bonus"] == 0.01


def test_continuity_is_additive_and_cannot_rescue_unscored_pair() -> None:
    s = PairScores({("a", "b"): 0.5})
    adj = Continuity([{"a", "b", "c"}], bonus=0.2).adjusted(s)
    assert adj.get("a", "b") == pytest.approx(0.7)
    assert adj.get("a", "c") == -math.inf
    with pytest.raises(ValueError):
        Continuity([], bonus=-0.1)


def test_unscored_pair_is_never_co_grouped() -> None:
    # a-b, c-d scored; e unscored against everyone: e must be a leftover, not seated.
    s = PairScores({("a", "b"): 0.9, ("c", "d"): 0.8, ("a", "c"): 0.1})
    result = assemble_groups(["a", "b", "c", "d", "e"], s, group_size=2)
    assert "e" in result.leftovers
    for g in result.groups:
        assert group_score(g, s) > -math.inf
    # a group of 3 needs three scored pairs; only a-b-c has them
    s3 = PairScores({("a", "b"): 0.9, ("b", "c"): 0.8, ("a", "c"): 0.7, ("c", "d"): 1.0})
    r3 = assemble_groups(["a", "b", "c", "d"], s3, group_size=3)
    assert r3.groups == (("a", "b", "c"),)
    assert r3.leftovers == ("d",)


def test_no_scored_pairs_means_no_groups() -> None:
    result = assemble_groups(["a", "b", "c"], PairScores(), group_size=2)
    assert result.groups == ()
    assert result.leftovers == ("a", "b", "c")
    assert result.global_min == math.inf
    assert result.audit == ()


def test_n_groups_caps_groups_formed() -> None:
    a, b, scores, _ = planted_two_cliques()
    result = assemble_groups([*a, *b], scores, group_size=6, n_groups=1)
    assert len(result.groups) == 1
    assert len(result.leftovers) == 6


def test_deterministic_under_seed() -> None:
    rng = random.Random(11)
    users = [f"u{i}" for i in range(18)]
    s = PairScores({k: rng.random() for k in complete_scores(users, 0.0)})
    first = assemble_groups(users, s, group_size=6, seed=5)
    second = assemble_groups(users, s, group_size=6, seed=5)
    assert first == second
    other = assemble_groups(users, s, group_size=6, seed=6)
    assert other.global_min > -math.inf
    assert {u for g in other.groups for u in g} | set(other.leftovers) == set(users)


def test_validates_arguments() -> None:
    with pytest.raises(ValueError):
        assemble_groups(["a"], PairScores(), group_size=1)
    with pytest.raises(ValueError):
        assemble_groups(["a"], PairScores(), iterations=-1)
    with pytest.raises(ValueError):
        assemble_groups(["a"], PairScores(), n_groups=-1)
