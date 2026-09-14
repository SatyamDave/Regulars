"""``python -m regulars.judge.cli known-pairs features.jsonl``

Each line of the JSONL file is one known pair::

    {"a": "priya", "b": "sam", "features_ab": {...}, "features_ba": {...}, "label": 1}

The default judge (``DEFAULT_WEIGHTS``) is applied with a z-scaler fit on the file's own
directed features, and the §6 report (variance, AUC, Spearman, ranking) is printed.
Pass ``--judge judge.json`` to use a serialised judge (its scaler included) instead.
Exit status is 1 when the report says the harness is broken (degenerate variance or
AUC below ``AUC_BROKEN``), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from regulars.core.features import PairFeatures
from regulars.judge.diagnostics import KnownPair, KnownPairsReport, known_pairs_report
from regulars.judge.v1 import JudgeV1


def load_known_pairs(path: Path) -> list[KnownPair]:
    pairs: list[KnownPair] = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                a, b = str(row["a"]), str(row["b"])
                label = int(row["label"])
                fab = {str(k): float(v) for k, v in row["features_ab"].items()}
                fba = {str(k): float(v) for k, v in row["features_ba"].items()}
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{path}:{lineno}: malformed known-pair row: {exc}") from exc
            if label not in (0, 1):
                raise ValueError(f"{path}:{lineno}: label must be 0 or 1, got {label}")
            pairs.append(
                KnownPair(
                    a=a,
                    b=b,
                    features_ab=PairFeatures(user_a=a, user_b=b, features=fab),
                    features_ba=PairFeatures(user_a=b, user_b=a, features=fba),
                    label=label,
                )
            )
    return pairs


def run_known_pairs(path: Path, judge_path: Path | None = None) -> KnownPairsReport:
    pairs = load_known_pairs(path)
    if judge_path is not None:
        judge = JudgeV1.from_json(judge_path.read_text())
    else:
        population = [pf for p in pairs for pf in (p.features_ab, p.features_ba)]
        judge = JudgeV1.default(population)
    return known_pairs_report(judge, pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.judge.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    kp = sub.add_parser("known-pairs", help="score a known-pairs JSONL file and print the report")
    kp.add_argument("features", type=Path, help="JSONL of {a,b,features_ab,features_ba,label}")
    kp.add_argument("--judge", type=Path, default=None, help="serialised JudgeV1 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "known-pairs":
        report = run_known_pairs(args.features, args.judge)
        print(report.render())
        return 1 if report.broken else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
