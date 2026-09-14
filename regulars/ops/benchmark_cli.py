"""``python -m regulars.ops.benchmark_cli assembly [--sizes ...] [--iterations N]``.

Diagnostic tooling, not part of the hot path: prints a wall-clock/memory table for
``regulars.assembly.benchmark.run_benchmark`` so the group-assembly optimizer's scaling
(technical-architecture.md §7, §8) can be checked by hand ahead of a population size that would
make it slow in production.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from regulars.assembly.benchmark import render_table, run_benchmark


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not sizes:
        raise ValueError("--sizes must list at least one population size")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.ops.benchmark_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    assembly = sub.add_parser("assembly", help="benchmark regulars.assembly.assemble_groups")
    assembly.add_argument(
        "--sizes",
        type=str,
        default="50,200,500,1000,2000",
        help="comma-separated population sizes, e.g. 50,200,500,1000",
    )
    assembly.add_argument(
        "--iterations", type=int, default=200, help="local-search iteration cap (default 200)"
    )
    assembly.add_argument(
        "--avg-candidates-per-user",
        type=int,
        default=200,
        help="target scored partners per user, mirroring the §8 prefilter cap (default 200)",
    )
    assembly.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "assembly":  # pragma: no cover - argparse enforces the choice
        return 2
    sizes = _parse_sizes(args.sizes)
    results = run_benchmark(
        sizes=sizes,
        avg_candidates_per_user=args.avg_candidates_per_user,
        iterations=args.iterations,
        seed=args.seed,
    )
    sys.stdout.write(render_table(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
