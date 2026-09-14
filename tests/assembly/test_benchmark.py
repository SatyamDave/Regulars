"""Fast regression test for regulars/assembly/benchmark.py.

See docs/perf/assembly-optimizer-benchmark.md for the full findings. Runs the benchmark
harness at small population sizes only (50 and 200 users) so this stays a
sub-second CI test, but the wall-clock budget below is tight enough that a future change making
``assemble_groups`` even 10-20x slower at these sizes fails it — see the perf doc for the actual
measured numbers up to 2000 users, which take too long to run on every CI invocation.
"""

from __future__ import annotations

import math

import pytest

from regulars.assembly.benchmark import (
    BenchmarkResult,
    generate_synthetic_population,
    planted_clique_min,
    render_table,
    run_benchmark,
)

# Measured on the reference machine: 50 users ~0.007s, 200 users ~0.14s (combined ~0.15s).
# 10x that margin comfortably absorbs CI noise/slower hardware while still catching a
# regression that makes the optimizer even an order of magnitude slower than today.
WALL_CLOCK_BUDGET_S = 8.0


def test_generate_synthetic_population_shape() -> None:
    users, scores = generate_synthetic_population(50, avg_candidates_per_user=20, seed=0)
    assert len(users) == 50
    assert len(set(users)) == 50
    # every user should have roughly avg_candidates_per_user scored partners (its clique of 5
    # plus noise pairs), not the full O(n) fan-out a naive all-pairs population would have. Noise
    # edges are proposed independently by both endpoints, so individual degree varies around the
    # target rather than hitting it exactly -- bound generously rather than exactly.
    degree: dict[str, int] = {u: 0 for u in users}
    for (a, b), _value in scores.pairs():
        degree[a] += 1
        degree[b] += 1
    assert all(0 < d <= 40 for d in degree.values())
    avg_degree = sum(degree.values()) / len(degree)
    assert 10 <= avg_degree <= 30


def test_generate_synthetic_population_deterministic() -> None:
    users1, scores1 = generate_synthetic_population(50, 20, seed=7)
    users2, scores2 = generate_synthetic_population(50, 20, seed=7)
    assert users1 == users2
    assert dict(scores1.pairs()) == dict(scores2.pairs())


def test_benchmark_completes_within_budget_and_recovers_planted_min() -> None:
    sizes = [50, 200]
    results = run_benchmark(sizes=sizes, avg_candidates_per_user=200, iterations=200, seed=0)

    assert [r.n_users for r in results] == sizes
    total_wall_time = sum(r.wall_time_s for r in results)
    assert total_wall_time < WALL_CLOCK_BUDGET_S, (
        f"assemble_groups took {total_wall_time:.3f}s combined at sizes {sizes}, "
        f"budget is {WALL_CLOCK_BUDGET_S}s -- looks like a real regression, not noise"
    )

    for size, result in zip(sizes, results, strict=True):
        planted = planted_clique_min(size, avg_candidates_per_user=200, seed=0)
        # The synthetic population's cliques score well above the noise pairs, so a working
        # optimizer should recover a global_min close to (and no better than) the planted value.
        assert result.global_min <= planted + 1e-9
        assert result.global_min == pytest.approx(planted, abs=0.05)


def test_benchmark_memory_estimate_is_finite_and_positive() -> None:
    results = run_benchmark(sizes=[50], avg_candidates_per_user=20, iterations=50, seed=0)
    (result,) = results
    assert math.isfinite(result.memory_estimate_mb)
    assert result.memory_estimate_mb > 0


def test_render_table_contains_header_and_rows() -> None:
    results = [
        BenchmarkResult(
            n_users=50,
            wall_time_s=0.01,
            global_min=0.7,
            per_group_min_p50=0.72,
            memory_estimate_mb=0.2,
        ),
        BenchmarkResult(
            n_users=200,
            wall_time_s=0.1,
            global_min=0.69,
            per_group_min_p50=0.71,
            memory_estimate_mb=3.9,
        ),
    ]
    table = render_table(results)
    assert "n_users" in table
    assert "wall_time_s" in table
    assert "50" in table
    assert "200" in table
    assert table.count("\n") == len(results) + 2  # header + separator + one row per result


def test_generate_synthetic_population_rejects_tiny_inputs() -> None:
    with pytest.raises(ValueError, match="n_users"):
        generate_synthetic_population(3, 10)
    with pytest.raises(ValueError, match="avg_candidates_per_user"):
        generate_synthetic_population(50, 1)
