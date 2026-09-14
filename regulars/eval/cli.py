"""``python -m regulars.eval.cli gate dyads.jsonl memberships.jsonl [options]``.

Prints the week-6 gate as Markdown. Attendance, RSVPs and all-pairs scores are optional
JSONL inputs; metrics without their inputs report ``insufficient_data`` rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from regulars.eval.gate import render_markdown, week6_gate
from regulars.eval.records import Attendance, LabeledDyad, Membership, Rsvp, load_jsonl


def _load_yhats(path: Path) -> list[float]:
    """One score per line: either a bare number or an object with a ``y_hat`` key."""
    out: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            out.append(float(obj["y_hat"] if isinstance(obj, dict) else obj))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.eval.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate", help="render the week-6 gate report")
    gate.add_argument("dyads", type=Path, help="LabeledDyad JSONL")
    gate.add_argument("memberships", type=Path, help="Membership JSONL")
    gate.add_argument("--attendance", type=Path, help="Attendance JSONL")
    gate.add_argument("--rsvps", type=Path, help="Rsvp JSONL")
    gate.add_argument(
        "--yhats",
        type=Path,
        help="all scored pairs' y_hat, one per line; defaults to the dyads' own y_hat",
    )
    gate.add_argument("--srm-r2", type=float, default=None, help="SRM relationship r²")
    gate.add_argument("--srm-p", type=float, default=None, help="p-value for --srm-r2")
    gate.add_argument("--reps", type=int, default=2000, help="bootstrap replicates")
    gate.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "gate":  # pragma: no cover - argparse enforces the choice
        return 2
    dyads = load_jsonl(args.dyads, LabeledDyad)
    memberships = load_jsonl(args.memberships, Membership)
    attendance = load_jsonl(args.attendance, Attendance) if args.attendance else []
    rsvps = load_jsonl(args.rsvps, Rsvp) if args.rsvps else []
    if args.yhats:
        y_hats = _load_yhats(args.yhats)
        fallback_note = ""
    else:
        y_hats = [d.y_hat for d in dyads]
        fallback_note = "yhat_std_all_pairs uses the labeled dyads only (--yhats not given)."
    report = week6_gate(
        dyads,
        memberships,
        attendance,
        rsvps,
        y_hats,
        srm_relationship_r2=args.srm_r2,
        srm_relationship_p=args.srm_p,
        reps=args.reps,
        seed=args.seed,
    )
    text = render_markdown(report)
    if fallback_note:
        text += f"- {fallback_note}\n"
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
