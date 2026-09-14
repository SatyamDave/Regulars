import numpy as np
import pytest

from regulars.judge.srm import (
    DirectedRating,
    fit_srm,
    group_effects,
    relationship_residuals,
    variance_explained_by,
)

N_GROUPS, GROUP_SIZE = 40, 6
ACTOR_VAR, PARTNER_VAR, REL_VAR = 1.0, 0.5, 0.25


def simulate(
    rng: np.random.Generator,
    n_groups: int = N_GROUPS,
    n: int = GROUP_SIZE,
    actor_var: float = ACTOR_VAR,
    partner_var: float = PARTNER_VAR,
    rel_var: float = REL_VAR,
    mu: float = 3.0,
) -> tuple[list[DirectedRating], dict[tuple[str, str, str], dict[str, float]]]:
    """Round-robins from known variance components; also returns the true effects per dyad."""
    ratings: list[DirectedRating] = []
    truth: dict[tuple[str, str, str], dict[str, float]] = {}
    for gi in range(n_groups):
        gid = f"g{gi}"
        a = rng.normal(0, np.sqrt(actor_var), n)
        b = rng.normal(0, np.sqrt(partner_var), n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                g = rng.normal(0, np.sqrt(rel_var)) if rel_var > 0 else 0.0
                ratings.append(DirectedRating(gid, f"p{i}", f"p{j}", mu + a[i] + b[j] + g))
                truth[(gid, f"p{i}", f"p{j}")] = {"actor": a[i], "partner": b[j], "rel": g}
    return ratings, truth


def test_recovers_known_variance_components() -> None:
    ratings, _ = simulate(np.random.default_rng(1234))
    res = fit_srm(ratings)
    assert res.n_groups == N_GROUPS
    assert res.n_dyads == N_GROUPS * GROUP_SIZE * (GROUP_SIZE - 1)
    # sampling SE at 40x6 is roughly 0.1 / 0.05 / 0.015 for the three components
    assert res.actor_var == pytest.approx(ACTOR_VAR, abs=0.25)
    assert res.partner_var == pytest.approx(PARTNER_VAR, abs=0.15)
    assert res.relationship_var == pytest.approx(REL_VAR, abs=0.06)
    assert res.actor_partner_cov == pytest.approx(0.0, abs=0.2)
    assert res.relationship_reciprocity_cov == pytest.approx(0.0, abs=0.05)
    assert sum(res.shares.values()) == pytest.approx(1.0)
    assert res.shares["actor"] > res.shares["partner"] > res.shares["relationship"]


def test_estimators_are_unbiased_over_many_groups() -> None:
    ratings, _ = simulate(np.random.default_rng(7), n_groups=2000)
    res = fit_srm(ratings)
    assert res.actor_var == pytest.approx(ACTOR_VAR, abs=0.05)
    assert res.partner_var == pytest.approx(PARTNER_VAR, abs=0.03)
    assert res.relationship_var == pytest.approx(REL_VAR, abs=0.01)


def test_zero_relationship_variance_is_recovered_as_zero() -> None:
    ratings, _ = simulate(np.random.default_rng(3), rel_var=0.0)
    res = fit_srm(ratings)
    assert res.relationship_var == pytest.approx(0.0, abs=1e-9)
    assert res.relationship_reciprocity_cov == pytest.approx(0.0, abs=1e-9)
    assert res.shares["relationship"] == pytest.approx(0.0, abs=1e-9)
    assert res.actor_var == pytest.approx(ACTOR_VAR, abs=0.25)


def test_effects_and_residuals_satisfy_srm_constraints() -> None:
    ratings, _ = simulate(np.random.default_rng(5), n_groups=3)
    for e in group_effects(ratings).values():
        assert sum(e.actor.values()) == pytest.approx(0.0, abs=1e-9)
        assert sum(e.partner.values()) == pytest.approx(0.0, abs=1e-9)
        for m in e.members:
            assert sum(v for (r, _), v in e.relationship.items() if r == m) == pytest.approx(
                0.0, abs=1e-9
            )
            assert sum(v for (_, t), v in e.relationship.items() if t == m) == pytest.approx(
                0.0, abs=1e-9
            )


def test_relationship_residuals_track_true_dyadic_effects() -> None:
    ratings, truth = simulate(np.random.default_rng(11), rel_var=1.0, actor_var=0.2)
    resid = relationship_residuals(ratings)
    assert set(resid) == set(truth)
    est = np.array([resid[k] for k in truth])
    true = np.array([truth[k]["rel"] for k in truth])
    assert np.corrcoef(est, true)[0, 1] > 0.8


def test_variance_explained_separates_charisma_detector_from_matching_engine() -> None:
    # Relationship noise leaks into the *estimated* actor/partner effects by
    # MS_sym/(2(n-2)) + MS_asym/(2n); at rel_var=0.25, n=6 the estimated partner effect has
    # reliability ~0.9, so a perfect charisma detector can reach r² ~0.9, not 1.0.
    ratings, truth = simulate(np.random.default_rng(21))
    rng = np.random.default_rng(22)
    # a "charisma detector": y_hat is the ratee's true partner effect plus noise
    charisma = {k: t["partner"] + rng.normal(0, 0.1) for k, t in truth.items()}
    ve = variance_explained_by(charisma, ratings)
    assert ve["partner"] > 0.8
    assert ve["relationship"] < 0.1
    assert ve["actor"] < 0.1
    # a "matching engine": y_hat tracks the true dyad-specific effect
    matcher = {k: t["rel"] + rng.normal(0, 0.1) for k, t in truth.items()}
    ve = variance_explained_by(matcher, ratings)
    assert ve["relationship"] > 0.7
    assert ve["partner"] < 0.1


def test_variance_explained_requires_every_dyad() -> None:
    ratings, truth = simulate(np.random.default_rng(1), n_groups=1)
    partial = {k: 0.0 for k in list(truth)[:-1]}
    with pytest.raises(ValueError, match="y_hat missing"):
        variance_explained_by(partial, ratings)


def test_rejects_incomplete_small_or_invalid_round_robins() -> None:
    ratings, _ = simulate(np.random.default_rng(1), n_groups=1)
    with pytest.raises(ValueError, match="not a complete round-robin"):
        fit_srm(ratings[:-1])
    with pytest.raises(ValueError, match="needs >= 4"):
        fit_srm(
            [
                DirectedRating("g", a, b, 1.0)
                for a in ("x", "y", "z")
                for b in ("x", "y", "z")
                if a != b
            ]
        )
    with pytest.raises(ValueError, match="self-rating"):
        fit_srm([*ratings, DirectedRating("g0", "p0", "p0", 1.0)])
    with pytest.raises(ValueError, match="duplicate"):
        fit_srm([*ratings, ratings[0]])
    with pytest.raises(ValueError, match="no ratings"):
        fit_srm([])


def test_pooling_is_weighted_by_group_size() -> None:
    rng = np.random.default_rng(9)
    small, _ = simulate(rng, n_groups=1, n=4, actor_var=1.0, partner_var=0.0, rel_var=0.0)
    large, _ = simulate(rng, n_groups=1, n=8, actor_var=1.0, partner_var=0.0, rel_var=0.0)
    large = [DirectedRating("big", r.rater, r.ratee, r.value) for r in large]
    per_group = {gid: e.components["actor_var"] for gid, e in group_effects(small + large).items()}
    pooled = fit_srm(small + large).actor_var
    assert pooled == pytest.approx((4 * per_group["g0"] + 8 * per_group["big"]) / 12)
