"""Max-min group assembly (technical-architecture.md §7).

Group score is the **minimum** pair score in the group, never the mean: one bad dyad kills a
table of six. Partitioning into groups maximising the minimum is NP-hard; at our scale the
heuristic the spec prescribes is enough:

1. Greedy seed: the highest-scoring available pair; grow the group one member at a time,
   picking whichever candidate leaves the resulting minimum highest.
2. Local search: swap members between groups (and with leftovers) while the objective improves,
   for at most ``iterations`` rounds. Each round applies the single best improving swap
   (steepest ascent) and stops early when none exists.

The objective compared during local search is the vector of per-group minimums sorted
ascending, compared lexicographically. Its first element is the global minimum, so "global min
improves" is always accepted; ties on the global minimum are broken by improving the next-worst
group, which lets the search keep working on tables other than the current worst one.

**Unscored pairs.** ``PairScores.get`` returns ``-inf`` for any pair it has no score for, and
hard-constraint rejections are expressed by simply not scoring the pair. A group containing such
a pair scores ``-inf`` and the optimizer never forms one: seeding, growth and swaps all refuse
moves that would create a ``-inf`` group. Users who cannot be placed in a fully scored group stay
in ``leftovers``.

**Continuity bonus (the only additive term).** Everything else in this module is a filter or a
minimum. The single exception is :class:`Continuity`: when supplied, ``bonus`` is *added* to the
score of every pair that shared a group in the previous round before the minimum is taken. Its
purpose is tie-breaking toward a converging group (§7: the product thesis is a converging group,
not a fresh table; without this the optimizer churns membership every week). It is deliberately
small by default so it re-orders near-ties and does not rescue a genuinely bad pairing, and it
can never lift an unscored pair: ``-inf + bonus == -inf``. Reported ``global_min`` and
``per_group_min`` are always the *raw* scores, without the bonus, so what is stored is what the
judge predicted.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from regulars.assembly.audit import AuditRecord, json_score, members_key
from regulars.assembly.prefilter import Pair, pair_key

DEFAULT_CONTINUITY_BONUS = 0.05


class PairScores:
    """Symmetric ``(a, b) -> score`` lookup. Missing pairs read as ``-inf``."""

    __slots__ = ("_scores",)

    def __init__(self, scores: Mapping[tuple[str, str], float] | None = None) -> None:
        self._scores: dict[Pair, float] = {}
        for (a, b), value in (scores or {}).items():
            self.set(a, b, value)

    def set(self, a: str, b: str, value: float) -> None:
        key = pair_key(a, b)
        existing = self._scores.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting scores for pair {key}: {existing} vs {value}")
        self._scores[key] = float(value)

    def get(self, a: str, b: str) -> float:
        return self._scores.get(pair_key(a, b), -math.inf)

    def has(self, a: str, b: str) -> bool:
        return pair_key(a, b) in self._scores

    def pairs(self) -> Iterable[tuple[Pair, float]]:
        return self._scores.items()

    def __len__(self) -> int:
        return len(self._scores)

    def masked(self, allowed: Collection[Pair]) -> PairScores:
        """Only the scores whose pair is in ``allowed`` (e.g. the prefilter's plausible set)."""
        out = PairScores()
        for key, value in self._scores.items():
            if key in allowed:
                out._scores[key] = value
        return out


def group_score(members: Iterable[str], scores: PairScores) -> float:
    """Minimum pair score over all pairs in ``members``. ``inf`` for fewer than two members."""
    ms = list(members)
    best = math.inf
    for i, a in enumerate(ms):
        for b in ms[i + 1 :]:
            s = scores.get(a, b)
            if s < best:
                best = s
                if best == -math.inf:
                    return best
    return best


class Continuity:
    """Previous-round co-membership; adds ``bonus`` to pairs that were seated together."""

    __slots__ = ("_prev", "bonus")

    def __init__(
        self,
        previous_groups: Iterable[Collection[str]],
        bonus: float = DEFAULT_CONTINUITY_BONUS,
    ) -> None:
        if bonus < 0:
            raise ValueError("continuity bonus must be non-negative")
        self.bonus = float(bonus)
        self._prev: dict[str, int] = {}
        for i, group in enumerate(previous_groups):
            for user in group:
                self._prev[user] = i

    def same_previous_group(self, a: str, b: str) -> bool:
        ia = self._prev.get(a)
        return ia is not None and ia == self._prev.get(b)

    def adjusted(self, scores: PairScores) -> PairScores:
        """``scores`` with ``bonus`` added to every previously co-grouped pair."""
        out = PairScores()
        for (a, b), value in scores.pairs():
            out._scores[(a, b)] = value + self.bonus if self.same_previous_group(a, b) else value
        return out


@dataclass(frozen=True)
class AssemblyResult:
    groups: tuple[tuple[str, ...], ...]
    leftovers: tuple[str, ...]
    global_min: float
    """Raw minimum pair score across all groups; ``inf`` when no group was formed."""
    per_group_min: tuple[float, ...]
    trace: tuple[str, ...]
    audit: tuple[AuditRecord, ...]


def _grow(
    group: list[str],
    pool: Sequence[str],
    obj: PairScores,
    rank: Mapping[str, int],
    group_size: int,
) -> bool:
    """Extend ``group`` in place to ``group_size``; False if no finite candidate exists."""
    in_group = set(group)
    current = group_score(group, obj)
    while len(group) < group_size:
        best: str | None = None
        best_min = -math.inf
        for c in pool:
            if c in in_group:
                continue
            m = current
            for g in group:
                s = obj.get(c, g)
                if s < m:
                    m = s
                    if m == -math.inf:
                        break
            if m == -math.inf:
                continue
            if best is None or m > best_min or (m == best_min and rank[c] < rank[best]):
                best, best_min = c, m
        if best is None:
            return False
        group.append(best)
        in_group.add(best)
        current = best_min
    return True


def assemble_groups(
    users: Iterable[str],
    scores: PairScores,
    group_size: int = 6,
    n_groups: int | None = None,
    continuity: Continuity | None = None,
    iterations: int = 200,
    seed: int = 0,
) -> AssemblyResult:
    """Partition ``users`` into groups of ``group_size`` maximising the minimum pair score.

    Deterministic for a given ``seed``: the seed fixes a tie-breaking order over users and
    nothing else is random. ``n_groups`` caps the number of groups formed (default: as many as
    the users allow). See the module docstring for the algorithm and the continuity bonus.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if n_groups is not None and n_groups < 0:
        raise ValueError("n_groups must be non-negative")

    order = sorted(set(users))
    random.Random(seed).shuffle(order)
    rank = {u: i for i, u in enumerate(order)}
    obj = continuity.adjusted(scores) if continuity is not None else scores

    target = (
        len(order) // group_size if n_groups is None else min(n_groups, len(order) // group_size)
    )
    pool: list[str] = list(order)
    groups: list[list[str]] = []
    trace: list[str] = []
    dead_seeds: set[Pair] = set()

    # --- greedy seeding -----------------------------------------------------------------------
    while len(groups) < target and len(pool) >= group_size:
        seed_pair: Pair | None = None
        seed_score = -math.inf
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                s = obj.get(a, b)
                if s == -math.inf or pair_key(a, b) in dead_seeds:
                    continue
                if seed_pair is None or s > seed_score:
                    seed_pair, seed_score = (a, b), s
        if seed_pair is None:
            trace.append("seed: no scored pair left among remaining users")
            break
        group = list(seed_pair)
        if _grow(group, pool, obj, rank, group_size):
            groups.append(group)
            pool = [u for u in pool if u not in set(group)]
            trace.append(
                f"seed {seed_pair[0]}+{seed_pair[1]} ({seed_score:.3f}) -> group "
                f"{members_key(group)} min={group_score(group, obj):.3f}"
            )
        else:
            dead_seeds.add(pair_key(*seed_pair))
            trace.append(f"seed {seed_pair[0]}+{seed_pair[1]} could not grow to {group_size}")

    # --- local search -------------------------------------------------------------------------
    mins = [group_score(g, obj) for g in groups]
    swaps_applied = 0
    for it in range(iterations if groups else 0):
        current = tuple(sorted(mins))
        best_vec = current
        best_move: tuple[int, int, int, int] | None = None  # (gi, ai, gj|-1, bj|pool index)

        def vec_with(i: int, mi: float, j: int | None, mj: float | None) -> tuple[float, ...]:
            trial = list(mins)
            trial[i] = mi
            if j is not None and mj is not None:
                trial[j] = mj
            return tuple(sorted(trial))

        for i, gi in enumerate(groups):
            for ai, a in enumerate(gi):
                # swap with a member of a later group
                for j in range(i + 1, len(groups)):
                    gj = groups[j]
                    for bj, b in enumerate(gj):
                        ni = gi[:ai] + [b] + gi[ai + 1 :]
                        mi = group_score(ni, obj)
                        if mi == -math.inf:
                            continue
                        nj = gj[:bj] + [a] + gj[bj + 1 :]
                        mj = group_score(nj, obj)
                        if mj == -math.inf:
                            continue
                        vec = vec_with(i, mi, j, mj)
                        if vec > best_vec:
                            best_vec, best_move = vec, (i, ai, j, bj)
                # swap with a leftover
                for pi, p in enumerate(pool):
                    ni = gi[:ai] + [p] + gi[ai + 1 :]
                    mi = group_score(ni, obj)
                    if mi == -math.inf:
                        continue
                    vec = vec_with(i, mi, None, None)
                    if vec > best_vec:
                        best_vec, best_move = vec, (i, ai, -1, pi)
        if best_move is None:
            trace.append(f"local search converged after {it} iteration(s)")
            break
        i, ai, j, bj = best_move
        a = groups[i][ai]
        if j >= 0:
            b = groups[j][bj]
            groups[i][ai], groups[j][bj] = b, a
            mins[j] = group_score(groups[j], obj)
            trace.append(f"swap {a}<->{b} (groups {i},{j}) min {current[0]:.3f}->{best_vec[0]:.3f}")
        else:
            b = pool[bj]
            groups[i][ai], pool[bj] = b, a
            trace.append(
                f"swap {a}<->{b} (group {i}, leftover) min {current[0]:.3f}->{best_vec[0]:.3f}"
            )
        mins[i] = group_score(groups[i], obj)
        swaps_applied += 1
    else:
        if groups:
            trace.append(f"local search hit the {iterations}-iteration cap")

    final_groups = tuple(members_key(g) for g in groups)
    per_group_min = tuple(group_score(g, scores) for g in final_groups)
    global_min = min(per_group_min) if per_group_min else math.inf
    leftovers = tuple(sorted(pool))
    audit = tuple(
        AuditRecord(
            decision_type="group_assembled",
            subject_user_id=None,
            group_members=g,
            reason=(
                "This table was put together by the model: every pairing passed the hard "
                f"filters and the weakest predicted pairing scores {m:.2f}."
            ),
            payload={
                "assembly_method": "model",
                "min_score": json_score(m),
                "group_size": group_size,
                "continuity_bonus": continuity.bonus if continuity is not None else 0.0,
                "seed": seed,
                "swaps_applied": swaps_applied,
            },
        )
        for g, m in zip(final_groups, per_group_min, strict=True)
    )
    return AssemblyResult(
        groups=final_groups,
        leftovers=leftovers,
        global_min=global_min,
        per_group_min=per_group_min,
        trace=tuple(trace),
        audit=audit,
    )
