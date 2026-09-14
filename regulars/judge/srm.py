"""Kenny's Social Relations Model for round-robin ratings (technical-architecture.md §5 "v2").

Each dinner is a natural round-robin: ``n`` people, everyone rates everyone, ``n(n-1)``
directed ratings. SRM decomposes a directed rating ``X_ij`` (``i`` rates ``j``) as::

    X_ij = mu + a_i + b_j + g_ij

- ``a_i``  actor effect:     how positively ``i`` rates others
- ``b_j``  partner effect:   how positively ``j`` is rated by others
- ``g_ij`` relationship:     the dyad-specific residual

This mirrors Joel/Eastwick/Finkel: actor and partner were predictable, relationship was not.
Fitting SRM tells you which component a model is actually predicting. A judge that only
recovers partner variance is a charisma detector, not a matching engine.

Estimators are the closed-form least-squares effects of Warner, Kenny & Stoto (1979) and the
unbiased variance-component formulas of Kenny (1994, *Interpersonal Perception*, ch. 9;
Kenny, Kashy & Cook 2006, ch. 8) for a **single round-robin without replication**. They are
the same formulas the R package ``TripleR`` (Schoenbrodt, Back & Schmukle) implements in
``RR()``. With no replication the relationship component is confounded with measurement
error, so ``relationship_var`` here is *relationship + error*; TripleR reports it the same
way.

For a group of size ``n`` with row means ``M_i.`` (given), column means ``M_.j`` (received)
and grand mean ``M..``::

    a_i  = (n-1)^2/(n(n-2)) * M_i. + (n-1)/(n(n-2)) * M_.i - (n-1)/(n-2) * M..
    b_j  = (n-1)^2/(n(n-2)) * M_.j + (n-1)/(n(n-2)) * M_j. - (n-1)/(n-2) * M..
    g_ij = X_ij - a_i - b_j - M..

    MS_sym  = sum_{i<j} ((g_ij + g_ji)/2)^2 * 2 / (n(n-3)/2)         estimates s2_g + s_gg
    MS_asym = sum_{i<j} ((g_ij - g_ji)/2)^2 * 2 / ((n-1)(n-2)/2)     estimates s2_g - s_gg

    relationship_var             = (MS_sym + MS_asym) / 2
    relationship_reciprocity_cov = (MS_sym - MS_asym) / 2
    actor_var         = sum a_i^2   / (n-1) - MS_sym/(2(n-2)) - MS_asym/(2n)
    partner_var       = sum b_j^2   / (n-1) - MS_sym/(2(n-2)) - MS_asym/(2n)
    actor_partner_cov = sum a_i b_i / (n-1) - MS_sym/(2(n-2)) + MS_asym/(2n)

The correction terms remove the leakage of relationship variance into the estimated effects;
they are what make the estimators unbiased rather than merely consistent. Group estimates are
pooled as a weighted mean with weight = group size, as ``TripleR`` does for multi-group data.
Variance estimates can come out negative on small samples; they are reported as estimated
(not clipped) and ``shares`` is computed on the clipped values.

Groups need ``n >= 4``: with ``n = 3`` the symmetric relationship term has zero degrees of
freedom and the model is not identified.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_GROUP_SIZE = 4

DyadKey = tuple[str, str, str]
"""``(group_id, rater, ratee)``."""


@dataclass(frozen=True, slots=True)
class DirectedRating:
    group_id: str
    rater: str
    ratee: str
    value: float


@dataclass(frozen=True, slots=True)
class SRMResult:
    actor_var: float
    partner_var: float
    relationship_var: float
    """Relationship **plus error**: a single round-robin without replication cannot separate
    the dyad-specific component from measurement noise."""
    actor_partner_cov: float
    relationship_reciprocity_cov: float
    n_groups: int
    n_dyads: int
    shares: dict[str, float]
    """Share of total (actor + partner + relationship) variance per component, computed on
    variance estimates clipped at zero."""


@dataclass(frozen=True, slots=True)
class GroupEffects:
    members: tuple[str, ...]
    grand_mean: float
    actor: dict[str, float]
    partner: dict[str, float]
    relationship: dict[tuple[str, str], float]
    components: dict[str, float]
    """Per-group variance components before pooling."""


def _matrices(ratings: Iterable[DirectedRating]) -> dict[str, dict[tuple[str, str], float]]:
    by_group: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for r in ratings:
        if r.rater == r.ratee:
            raise ValueError(f"self-rating in group {r.group_id!r}: {r.rater!r}")
        key = (r.rater, r.ratee)
        if key in by_group[r.group_id]:
            raise ValueError(f"duplicate rating in group {r.group_id!r}: {key}")
        by_group[r.group_id][key] = r.value
    return by_group


def _to_array(
    group_id: str, cells: Mapping[tuple[str, str], float]
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    members = tuple(sorted({p for pair in cells for p in pair}))
    n = len(members)
    if n < MIN_GROUP_SIZE:
        raise ValueError(f"group {group_id!r} has {n} members; SRM needs >= {MIN_GROUP_SIZE}")
    index = {m: k for k, m in enumerate(members)}
    x = np.full((n, n), np.nan, dtype=np.float64)
    for (rater, ratee), v in cells.items():
        x[index[rater], index[ratee]] = v
    off = ~np.eye(n, dtype=bool)
    if np.isnan(x[off]).any():
        missing = int(np.isnan(x[off]).sum())
        raise ValueError(f"group {group_id!r} is not a complete round-robin: {missing} missing")
    return members, x


def _decompose(members: tuple[str, ...], x: NDArray[np.float64]) -> GroupEffects:
    n = len(members)
    off = ~np.eye(n, dtype=bool)
    filled = np.where(off, x, 0.0)
    row_mean = filled.sum(axis=1) / (n - 1)
    col_mean = filled.sum(axis=0) / (n - 1)
    grand = float(x[off].mean())

    c1 = (n - 1) ** 2 / (n * (n - 2))
    c2 = (n - 1) / (n * (n - 2))
    c3 = (n - 1) / (n - 2)
    a = c1 * row_mean + c2 * col_mean - c3 * grand
    b = c1 * col_mean + c2 * row_mean - c3 * grand
    g = x - a[:, None] - b[None, :] - grand
    g[~off] = 0.0

    iu, ju = np.triu_indices(n, k=1)
    sym = (g[iu, ju] + g[ju, iu]) / 2.0
    asym = (g[iu, ju] - g[ju, iu]) / 2.0
    df_sym = n * (n - 3) / 2.0
    df_asym = (n - 1) * (n - 2) / 2.0
    ms_sym = 2.0 * float(np.sum(sym**2)) / df_sym
    ms_asym = 2.0 * float(np.sum(asym**2)) / df_asym

    rel_var = (ms_sym + ms_asym) / 2.0
    rel_cov = (ms_sym - ms_asym) / 2.0
    corr = ms_sym / (2.0 * (n - 2))
    actor_var = float(np.sum(a**2)) / (n - 1) - corr - ms_asym / (2.0 * n)
    partner_var = float(np.sum(b**2)) / (n - 1) - corr - ms_asym / (2.0 * n)
    ab_cov = float(np.sum(a * b)) / (n - 1) - corr + ms_asym / (2.0 * n)

    return GroupEffects(
        members=members,
        grand_mean=grand,
        actor={m: float(a[k]) for k, m in enumerate(members)},
        partner={m: float(b[k]) for k, m in enumerate(members)},
        relationship={
            (members[i], members[j]): float(g[i, j]) for i in range(n) for j in range(n) if i != j
        },
        components={
            "actor_var": actor_var,
            "partner_var": partner_var,
            "relationship_var": rel_var,
            "actor_partner_cov": ab_cov,
            "relationship_reciprocity_cov": rel_cov,
        },
    )


def group_effects(ratings: Iterable[DirectedRating]) -> dict[str, GroupEffects]:
    """Per-group SRM effects and components, keyed by ``group_id``."""
    out: dict[str, GroupEffects] = {}
    for gid, cells in _matrices(ratings).items():
        members, x = _to_array(gid, cells)
        out[gid] = _decompose(members, x)
    return out


def fit_srm(ratings: Iterable[DirectedRating]) -> SRMResult:
    """Pooled variance components across all groups (weight = group size)."""
    effects = group_effects(ratings)
    if not effects:
        raise ValueError("no ratings")
    weights = np.array([len(e.members) for e in effects.values()], dtype=np.float64)
    names = (
        "actor_var",
        "partner_var",
        "relationship_var",
        "actor_partner_cov",
        "relationship_reciprocity_cov",
    )
    pooled = {
        name: float(np.average([e.components[name] for e in effects.values()], weights=weights))
        for name in names
    }
    clipped = {k: max(pooled[k], 0.0) for k in ("actor_var", "partner_var", "relationship_var")}
    total = sum(clipped.values())
    shares = {
        "actor": clipped["actor_var"] / total if total > 0 else float("nan"),
        "partner": clipped["partner_var"] / total if total > 0 else float("nan"),
        "relationship": clipped["relationship_var"] / total if total > 0 else float("nan"),
    }
    return SRMResult(
        actor_var=pooled["actor_var"],
        partner_var=pooled["partner_var"],
        relationship_var=pooled["relationship_var"],
        actor_partner_cov=pooled["actor_partner_cov"],
        relationship_reciprocity_cov=pooled["relationship_reciprocity_cov"],
        n_groups=len(effects),
        n_dyads=sum(len(e.relationship) for e in effects.values()),
        shares=shares,
    )


def relationship_residuals(ratings: Iterable[DirectedRating]) -> dict[DyadKey, float]:
    """``g_ij`` per directed dyad, keyed ``(group_id, rater, ratee)``."""
    return {
        (gid, rater, ratee): g
        for gid, e in group_effects(ratings).items()
        for (rater, ratee), g in e.relationship.items()
    }


def _r2(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.std() == 0.0 or ya.std() == 0.0:
        return float("nan")
    r = float(np.corrcoef(xa, ya)[0, 1])
    return r * r


def variance_explained_by(
    y_hat_by_dyad: Mapping[DyadKey, float], ratings: Iterable[DirectedRating]
) -> dict[str, float]:
    """r² of each SRM component of the *ratings* against the same component of *y_hat*.

    ``y_hat`` is itself a directed round-robin (the judge scores a->b for every dyad), so it
    is decomposed with the same estimators and the components are compared like for like:
    actor effect of ``y_hat`` vs actor effect of the rating, and so on. Reading:

    - high ``partner`` r² only: the judge finds likeable people (charisma detector)
    - high ``actor`` r² only: the judge finds generous raters (rater bias)
    - ``relationship`` r² > 0: the judge predicts dyad-specific fit (matching engine)

    Every rated dyad must have a ``y_hat``; raises otherwise. Returns ``nan`` for a component
    that is constant on either side.
    """
    rating_list = list(ratings)
    truth = group_effects(rating_list)
    predicted_ratings = []
    for r in rating_list:
        key = (r.group_id, r.rater, r.ratee)
        if key not in y_hat_by_dyad:
            raise ValueError(f"y_hat missing for dyad {key}")
        predicted_ratings.append(DirectedRating(r.group_id, r.rater, r.ratee, y_hat_by_dyad[key]))
    pred = group_effects(predicted_ratings)

    actor_t: list[float] = []
    actor_p: list[float] = []
    partner_t: list[float] = []
    partner_p: list[float] = []
    rel_t: list[float] = []
    rel_p: list[float] = []
    for gid, te in truth.items():
        pe = pred[gid]
        for m in te.members:
            actor_t.append(te.actor[m])
            actor_p.append(pe.actor[m])
            partner_t.append(te.partner[m])
            partner_p.append(pe.partner[m])
        for dyad, g in te.relationship.items():
            rel_t.append(g)
            rel_p.append(pe.relationship[dyad])
    return {
        "actor": _r2(actor_p, actor_t),
        "partner": _r2(partner_p, partner_t),
        "relationship": _r2(rel_p, rel_t),
    }
