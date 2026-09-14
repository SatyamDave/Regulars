# `assemble_groups` scaling benchmark

Measured with `regulars/assembly/benchmark.py` (`python -m regulars.ops.benchmark_cli assembly`)
on the development machine (Apple Silicon, single core, CPython 3.12.13), `iterations=200`,
`avg_candidates_per_user=200` (the §8 prefilter cap), synthetic planted-clique populations,
`seed=0`. Numbers are a single run each; see "Noise" below for how much that matters.

## Results

| n_users | wall_time_s | global_min | per_group_min (p50) | memory_estimate_mb |
|---:|---:|---:|---:|---:|
| 50 | 0.0056 | 0.7003 | 0.7215 | 0.24 |
| 200 | 0.1242 | 0.7001 | 0.7105 | 3.69 |
| 500 | 0.6636 | 0.7001 | 0.7090 | 10.85 |
| 1000 | 4.1712 | 0.7001 | 0.7098 | 21.70 |
| 2000 | 37.8293 | 0.7000 | 0.7105 | 43.40 |

`global_min` recovers the planted clique minimum (~0.70) at every size, so the optimizer is
finding the intended optimum, not just running fast on garbage.

## Is it linear, quadratic, or worse?

Implied exponent between consecutive sizes (`log(time_ratio) / log(n_ratio)`):

| step | n ratio | time ratio | implied exponent |
|---|---:|---:|---:|
| 50 → 200 | 4.0x | 22.0x | 2.23 |
| 200 → 500 | 2.5x | 5.3x | 1.83 |
| 500 → 1000 | 2.0x | 6.3x | 2.65 |
| 1000 → 2000 | 2.0x | 9.1x | 3.18 |

Small-n noise makes the early ratios wobble, but the trend across the larger, more reliable
steps is unambiguous: the implied exponent is **climbing toward 3, not settling near 2**. This
is not the fixed-200-iteration local search, which is architecturally `O(iterations * n^2)` and
should show a flat exponent around 2 once `n` is large enough to dominate the constant. Something
else is asymptotically worse than quadratic.

### Root cause: cProfile at n=1000

```
104,136,918 function calls in 16.1s (cProfile overhead included, ~2.3x the perf_counter time)
30,840,530 calls to PairScores.get   (11.8s cumulative)
   502,354 calls to group_score       (1.0s cumulative, from local search)
       166 calls to _grow             (0.2s cumulative, from greedy seeding)
```

`_grow` is only called once per group formed (166 times for 1000 users / group_size 6) — cheap.
The 30.8M `PairScores.get` calls are dominated by the **greedy-seeding loop** in
`assemble_groups` (`regulars/assembly/optimizer.py:214-224`): for every group it forms, it does
a full double loop `for i, a in enumerate(pool): for b in pool[i+1:]:` over the *entire
remaining pool* to find the single best-scoring pair to seed the next group. Pool size shrinks
by only `group_size` (6) each time a group is formed, so the total work is

```
sum over k=0..n/6 of (n - 6k)^2  ≈  n^3 / 18
```

That is **cubic in the number of users being assembled in one round**, independent of
`iterations`. The 200-iteration local search is the quadratic piece (`O(iterations * n^2)`,
constant `iterations`) the spec was worried about, and it is *not* the bottleneck at these
sizes — the seeding loop is, and it gets relatively worse as `n` grows because a cubic term
eventually dominates any fixed-multiple quadratic term.

## Recommendation for §8's build order

At 200 candidates/user this benchmarks a single assembly round's *user count*, not the judge's
per-user candidate count — §8's prefilter already keeps the judge's cost linear in users. The
number this benchmark stresses is **how many users get assembled together in one batch/round**.

- **Up to ~500 users per assembly batch: fine as-is.** Well under a second; nowhere near a
  concern for a nightly or even hourly batch job.
- **~1,000 users per batch: ~4s. Still fine for a nightly batch,** but starting to be visible if
  assembly ever needs to run inline (e.g. a manual "re-assemble now" admin action) rather than
  as an offline job.
- **~2,000 users per batch: ~38s and rising cubically.** If Regulars ever assembles a single
  large market (a city, a large cohort) as one batch rather than in per-market or per-cohort
  chunks of a few hundred users, this becomes the real bottleneck, well before the judge's
  simulation cost that §8 was written to control. At 10,000 users in one batch the cubic term
  alone (ignoring even worse dict/GC overhead at that size) would extrapolate to roughly
  `38 * (10000/2000)^3 ≈ 4750s` (~80 minutes) for seeding alone — a genuine problem for any
  batch window, and the kind of "surprise bill" the cost-model section is about, just paid in
  compute time and job latency instead of judge API dollars.
- **Concrete fix if/when this becomes real:** the seeding loop does not need to rescan the whole
  remaining pool for a new best pair after every single group is formed. Two options, in
  increasing order of effort:
  1. Cap the seeding scan: seed from a bounded random or top-K sample of the remaining pool
     instead of an exhaustive scan (turns the seeding step into `O(n * K)` instead of `O(n^3)`,
     matching the sampling-based spirit already used elsewhere in `regulars/assembly`).
  2. Maintain a max-heap (or sorted structure) of remaining scored pairs, invalidating entries
     lazily as users get assigned to groups, so each seed pick is `O(log n)` amortized instead
     of `O(n^2)`.
  Either change is local to the greedy-seeding block in `assemble_groups`
  (`regulars/assembly/optimizer.py`) and does not need to touch the local-search loop, the
  public signature, or anything in `prefilter.py`/`exploration.py`.
- **Until batches regularly exceed ~1,000-2,000 users in one assembly round, this is not worth
  fixing.** The existing 200-iteration local search is correctly quadratic-and-bounded as the
  spec intended; the actual risk is the unbounded seeding scan, and it should be addressed if
  and when batch sizes approach the low thousands, not preemptively.

## Noise

Single-run measurements on a shared development machine; the wall-clock numbers can move
20-30% run to run, especially at the smaller sizes where `perf_counter` resolution and Python
startup/JIT-adjacent effects matter more. The *trend* (climbing implied exponent, cProfile call
counts dominated by the seeding double loop) is the load-bearing finding, not any single digit.
The regression test at `tests/assembly/test_benchmark.py` only asserts a generous combined
wall-clock budget at n=50/200 (default 8s vs. a measured ~0.13s combined — roughly 60x margin)
so it catches a real regression without being sensitive to this noise.
