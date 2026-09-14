"""Round-2 retention, model arm vs random arm (§6 Problem 1: the A/B test that matters).

Definition: among users who were members of a group in round 1, the share who attended one of
that group's round-2 plans. A group that never held a round 2 counts its round-1 members as not
retained: a dead group is the negative signal, not missing data. Callers should pass only
groups whose round-2 window has closed.

The CI on the difference comes from a cluster bootstrap over groups, matching
:mod:`regulars.eval.correlation`; retention is decided at the table, not per person.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from regulars.eval.records import AssemblyMethod, Attendance, Membership

ROUND_ONE = 1
ROUND_TWO = 2


@dataclass(frozen=True, slots=True)
class RetentionResult:
    model_rate: float
    random_rate: float
    diff_pp: float
    ci95: tuple[float, float]
    n_model: int
    n_random: int
    n_groups_model: int
    n_groups_random: int

    @property
    def n(self) -> int:
        return self.n_model + self.n_random

    @property
    def pooled_rate(self) -> float:
        if self.n == 0:
            return float("nan")
        return (self.model_rate * self.n_model + self.random_rate * self.n_random) / self.n


def _outcomes_by_group(
    memberships: Sequence[Membership], attendance: Sequence[Attendance]
) -> dict[str, tuple[AssemblyMethod, list[int]]]:
    """Per group: assembly method and one 0/1 per round-1 member (1 = attended round 2)."""
    attended_r2: set[tuple[str, str]] = {
        (a.group_id, a.user_id) for a in attendance if a.round_index == ROUND_TWO and a.attended
    }
    method_of: dict[str, AssemblyMethod] = {}
    members_r1: dict[str, set[str]] = defaultdict(set)
    for m in memberships:
        prior = method_of.setdefault(m.group_id, m.assembly_method)
        if prior != m.assembly_method:
            raise ValueError(f"group {m.group_id} has mixed assembly_method")
        if m.member_in_round(ROUND_ONE):
            members_r1[m.group_id].add(m.user_id)
    return {
        g: (method_of[g], [int((g, u) in attended_r2) for u in sorted(users)])
        for g, users in members_r1.items()
    }


def _rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def round2_retention(
    memberships: Sequence[Membership],
    attendance: Sequence[Attendance],
    reps: int = 2000,
    seed: int = 0,
) -> RetentionResult:
    by_group = _outcomes_by_group(memberships, attendance)
    model_groups = [v for m, v in by_group.values() if m == "model"]
    random_groups = [v for m, v in by_group.values() if m == "random"]
    model_flat = [x for g in model_groups for x in g]
    random_flat = [x for g in random_groups for x in g]
    model_rate = _rate(model_flat)
    random_rate = _rate(random_flat)
    diff_pp = 100.0 * (model_rate - random_rate)

    ci95: tuple[float, float] = (float("nan"), float("nan"))
    if len(model_groups) >= 2 and len(random_groups) >= 2:
        rng = np.random.default_rng(seed)
        diffs = np.empty(reps, dtype=np.float64)
        for r in range(reps):
            mi = rng.integers(0, len(model_groups), size=len(model_groups))
            ri = rng.integers(0, len(random_groups), size=len(random_groups))
            m_vals = [x for k in mi for x in model_groups[k]]
            r_vals = [x for k in ri for x in random_groups[k]]
            diffs[r] = 100.0 * (float(np.mean(m_vals)) - float(np.mean(r_vals)))
        lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
        ci95 = (lo, hi)

    return RetentionResult(
        model_rate=model_rate,
        random_rate=random_rate,
        diff_pp=diff_pp,
        ci95=ci95,
        n_model=len(model_flat),
        n_random=len(random_flat),
        n_groups_model=len(model_groups),
        n_groups_random=len(random_groups),
    )
