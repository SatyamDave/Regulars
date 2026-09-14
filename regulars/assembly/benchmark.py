"""Standalone benchmark harness for :func:`assemble_groups` (technical-architecture.md §7, §8).

§8 says prefiltering keeps the *judge*'s simulation cost linear (~200 candidates per user
instead of all-pairs), but nobody has measured how the greedy-seed + local-search optimizer
itself scales once the number of users fed into one assembly round grows toward the thousands
§8 describes. This module is diagnostic tooling only: it is never imported by the assembly
package itself, has no dependency on pytest, and generates its own synthetic population instead
of touching the judge, the DB, or any LLM.

``generate_synthetic_population`` builds a "planted clique" style population: users are split
into cliques of ``group_size`` with high, correlated in-clique scores plus low-amplitude noise,
and every other pair gets uniform noise. That gives ``assemble_groups`` a reachable optimum
(useful for a sanity check on ``global_min``) while still producing a realistic-looking, mostly
sparse candidate graph once capped to ``avg_candidates_per_user``.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass

from regulars.assembly.optimizer import PairScores, assemble_groups

GROUP_SIZE = 6
CLIQUE_SCORE_RANGE = (0.70, 0.95)
NOISE_SCORE_RANGE = (0.00, 0.40)


def _build_population(
    n_users: int,
    avg_candidates_per_user: int,
    seed: int,
) -> tuple[list[str], PairScores, float]:
    """Build ``n_users`` users and a capped, planted-clique-style :class:`PairScores`.

    Users are chunked into cliques of ``GROUP_SIZE``; every in-clique pair scores high (with
    small per-pair jitter), every other candidate pair scores as noise. The candidate graph is
    then thinned so each user keeps roughly ``avg_candidates_per_user`` scored partners (its
    whole clique plus a random sample of noise pairs), mirroring the prefilter's per-user cap
    (§8) without depending on ``regulars.assembly.prefilter`` itself.

    Returns ``(users, scores, planted_min)`` where ``planted_min`` is the minimum in-clique
    score — the value a perfect optimizer run on the full (unthinned) clique scores would
    recover as ``global_min``. Deterministic for a given ``seed``, so callers that only need two
    of the three values can call this (or the public wrappers below) repeatedly and get
    identical results.
    """
    if n_users < GROUP_SIZE:
        raise ValueError(f"n_users must be at least {GROUP_SIZE}")
    if avg_candidates_per_user < GROUP_SIZE - 1:
        raise ValueError(f"avg_candidates_per_user must be at least {GROUP_SIZE - 1}")

    rng = random.Random(seed)
    users = [f"u{i:06d}" for i in range(n_users)]
    n_cliques = n_users // GROUP_SIZE
    cliques = [users[i * GROUP_SIZE : (i + 1) * GROUP_SIZE] for i in range(n_cliques)]

    raw: dict[tuple[str, str], float] = {}
    planted_min = 1.0
    for clique in cliques:
        for i, a in enumerate(clique):
            for b in clique[i + 1 :]:
                value = rng.uniform(*CLIQUE_SCORE_RANGE)
                raw[(a, b)] = value
                planted_min = min(planted_min, value)

    # Extra noise candidates per user, capped so the graph stays sparse like a prefiltered one.
    # Halved because each side of a pair independently proposes noise edges: a user's final
    # degree is roughly its own proposals plus the (similar-sized) count of other users who
    # happened to propose it, so proposing at half the target degree lands close to the target.
    extra_per_user = max(0, (avg_candidates_per_user - (GROUP_SIZE - 1)) // 2)
    for a in users:
        others = [u for u in users if u != a]
        rng.shuffle(others)
        added = 0
        for b in others:
            if added >= extra_per_user:
                break
            key = (a, b) if a < b else (b, a)
            if key in raw:
                continue
            raw[key] = rng.uniform(*NOISE_SCORE_RANGE)
            added += 1

    return users, PairScores(raw), planted_min if cliques else float("-inf")


def generate_synthetic_population(
    n_users: int,
    avg_candidates_per_user: int,
    seed: int = 0,
) -> tuple[list[str], PairScores]:
    """Public generator matching the harness contract: ``(users, scores)``.

    See :func:`_build_population` for the generation scheme and :func:`planted_clique_min` to
    recover the planted optimum for the same arguments (used by the regression test's sanity
    check, not by the timed benchmark path).
    """
    users, scores, _planted_min = _build_population(n_users, avg_candidates_per_user, seed)
    return users, scores


def planted_clique_min(n_users: int, avg_candidates_per_user: int, seed: int = 0) -> float:
    """The minimum in-clique score planted by :func:`generate_synthetic_population`.

    Recomputes the same deterministic population (same ``seed``) to recover the value a perfect
    optimizer would report as ``global_min``.
    """
    _users, _scores, planted_min = _build_population(n_users, avg_candidates_per_user, seed)
    return planted_min


@dataclass(frozen=True)
class BenchmarkResult:
    n_users: int
    wall_time_s: float
    global_min: float
    per_group_min_p50: float
    memory_estimate_mb: float


def _pair_scores_memory_mb(scores: PairScores) -> float:
    """Rough resident-size estimate for a :class:`PairScores`: no external deps required.

    Sums ``sys.getsizeof`` over the internal dict, its keys and its float values. This
    undercounts a little (e.g. string interning, dict overallocation) but is stable enough to
    show the memory trend across population sizes without adding ``tracemalloc`` bookkeeping to
    every timed call.
    """
    total = sys.getsizeof(scores._scores)  # noqa: SLF001 - benchmark harness inspects internals
    for (a, b), value in scores.pairs():
        total += sys.getsizeof((a, b)) + sys.getsizeof(a) + sys.getsizeof(b) + sys.getsizeof(value)
    return total / (1024 * 1024)


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def run_one(
    n_users: int,
    avg_candidates_per_user: int = 200,
    iterations: int = 200,
    seed: int = 0,
) -> BenchmarkResult:
    """Generate a population of ``n_users`` and time one :func:`assemble_groups` call."""
    users, scores, _planted_min = _build_population(n_users, avg_candidates_per_user, seed)
    start = time.perf_counter()
    result = assemble_groups(users, scores, group_size=GROUP_SIZE, iterations=iterations, seed=seed)
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        n_users=n_users,
        wall_time_s=elapsed,
        global_min=result.global_min,
        per_group_min_p50=_median(list(result.per_group_min)),
        memory_estimate_mb=_pair_scores_memory_mb(scores),
    )


def run_benchmark(
    sizes: list[int] | None = None,
    avg_candidates_per_user: int = 200,
    iterations: int = 200,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Time :func:`assemble_groups` at each population size in ``sizes`` (default 50..2000)."""
    if sizes is None:
        sizes = [50, 200, 500, 1000, 2000]
    return [
        run_one(
            n, avg_candidates_per_user=avg_candidates_per_user, iterations=iterations, seed=seed
        )
        for n in sizes
    ]


def render_table(results: list[BenchmarkResult]) -> str:
    """A fixed-width text table of benchmark results, human-readable on a terminal or in CI logs."""
    header = (
        f"{'n_users':>8} | {'wall_time_s':>11} | {'global_min':>10} | "
        f"{'per_group_p50':>13} | {'mem_est_mb':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.n_users:>8} | {r.wall_time_s:>11.4f} | {r.global_min:>10.4f} | "
            f"{r.per_group_min_p50:>13.4f} | {r.memory_estimate_mb:>10.3f}"
        )
    return "\n".join(lines) + "\n"
