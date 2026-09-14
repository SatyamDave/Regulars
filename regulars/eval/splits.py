"""Person-grouped cross-validation (§6 Problem 2: split by person, never by pair).

Users are assigned to folds; a dyad is *test* only when both users are in the test fold and
*train* only when both are in training folds. Dyads straddling the boundary are dropped from
that fold entirely, because either side of a straddler leaks one user's partner effect.

Cost: with ``k`` folds and users assigned uniformly, a share of roughly ``2/k · (1 - 1/k)`` of
dyads is dropped per fold (32% at ``k = 5``). ``FoldSplit.dropped_share`` reports the realized
loss so it is never silent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from regulars.eval.records import LabeledDyad


@dataclass(frozen=True, slots=True)
class FoldSplit:
    fold: int
    test_users: frozenset[str]
    train_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    dropped_idx: tuple[int, ...]

    @property
    def n_total(self) -> int:
        return len(self.train_idx) + len(self.test_idx) + len(self.dropped_idx)

    @property
    def dropped_share(self) -> float:
        return len(self.dropped_idx) / self.n_total if self.n_total else 0.0


def assign_users_to_folds(users: Sequence[str], n_folds: int, seed: int) -> dict[str, int]:
    """Seeded, balanced assignment of distinct user ids to ``n_folds`` folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique = sorted(set(users))
    if len(unique) < n_folds:
        raise ValueError(f"{len(unique)} users cannot fill {n_folds} folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    return {unique[int(i)]: int(k % n_folds) for k, i in enumerate(order)}


def person_grouped_folds(
    dyads: Sequence[LabeledDyad], n_folds: int = 5, seed: int = 0
) -> list[FoldSplit]:
    users = [d.rater for d in dyads] + [d.ratee for d in dyads]
    fold_of = assign_users_to_folds(users, n_folds, seed)
    splits: list[FoldSplit] = []
    for fold in range(n_folds):
        train: list[int] = []
        test: list[int] = []
        dropped: list[int] = []
        for i, d in enumerate(dyads):
            a, b = fold_of[d.rater] == fold, fold_of[d.ratee] == fold
            if a and b:
                test.append(i)
            elif not a and not b:
                train.append(i)
            else:
                dropped.append(i)
        splits.append(
            FoldSplit(
                fold=fold,
                test_users=frozenset(u for u, f in fold_of.items() if f == fold),
                train_idx=tuple(train),
                test_idx=tuple(test),
                dropped_idx=tuple(dropped),
            )
        )
    return splits


def users_of(dyads: Sequence[LabeledDyad], idx: Sequence[int]) -> set[str]:
    out: set[str] = set()
    for i in idx:
        out.add(dyads[i].rater)
        out.add(dyads[i].ratee)
    return out


def assert_no_leakage(dyads: Sequence[LabeledDyad], split: FoldSplit) -> None:
    """Raise if any user appears on both sides of ``split`` or the partitions overlap."""
    if set(split.train_idx) & set(split.test_idx):
        raise AssertionError(f"fold {split.fold}: train and test share dyad indices")
    shared = users_of(dyads, split.train_idx) & users_of(dyads, split.test_idx)
    if shared:
        sample = ", ".join(sorted(shared)[:5])
        raise AssertionError(f"fold {split.fold}: {len(shared)} users in train and test ({sample})")
    stray = users_of(dyads, split.test_idx) - split.test_users
    if stray:
        raise AssertionError(f"fold {split.fold}: test dyads touch non-test users {sorted(stray)}")
