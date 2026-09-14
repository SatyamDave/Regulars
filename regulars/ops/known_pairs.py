"""``python -m regulars.ops.known_pairs build --input pairs.yaml --output features.jsonl``

The missing upstream half of technical-architecture.md §5's build order step 1 ("Take 5 to 10
friend pairs where you already know the answer. Run extraction over their real chat history.").
``regulars.judge.cli known-pairs`` already scores a ``features.jsonl`` of pre-computed
``features_ab``/``features_ba``/``label`` rows; nothing before this module actually produced
that file from a real chat export. This module does, end to end, for free (heuristic
classifiers, zero API cost) by default.

Input format (``pairs.yaml``), a YAML list of::

    - a: priya
      b: sam
      label: 1
      messages: priya_sam.json

``messages`` points at a JSON file in one of two shapes, auto-detected:

* A BlueBubbles ``/api/v1/message`` export (a JSON list of raw payloads) — the shape
  :func:`regulars.features.ingest.message_row_from_bluebubbles` already parses. Detected by
  the first row carrying ``ROWID`` or ``dateCreated``. The export owner (``isFromMe``) is
  mapped to ``a``; the other party to ``b`` (BlueBubbles dyadic exports have exactly two
  senders, so no address book is needed).
* A generic JSON list for people exporting chat history by hand, without BlueBubbles::

    [{"sender": "priya", "text": "hey", "sent_at_ms": 1700000000000}, ...]

  ``sender`` must be exactly ``a`` or ``b``.

For each pair: messages -> :class:`~regulars.core.transcript.Transcript` (via
:func:`~regulars.features.ingest.transcript_from_messages`) -> shared features (via
:func:`~regulars.features.extract.extract_pair_features`, heuristic config by default,
``--llm-config frontier`` for :func:`~regulars.llm.factory.llm_from_env`-backed classifiers) ->
real-only features (via :func:`~regulars.features.real.extract_real_features`), merged in. The
real-only baseline is fit from this one conversation's own reply gaps, not a full message
history nobody has for a one-weekend exercise; treat ``latency_norm_*`` accordingly as noisier
than it would be with a real per-user baseline.

Output: one JSON line per pair, in exactly the shape ``regulars.judge.cli known-pairs`` reads
(``a``, ``b``, ``label``, ``features_ab``, ``features_ba``) — feed it straight in::

    python -m regulars.judge.cli known-pairs features.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

from regulars.core.transcript import Transcript
from regulars.features.extract import ExtractorConfig, extract_features_sync
from regulars.features.ingest import (
    MessageRow,
    message_row_from_bluebubbles,
    transcript_from_messages,
)
from regulars.features.real import extract_real_features, fit_baseline, response_latencies
from regulars.llm.factory import llm_from_env

LLMConfigName = Literal["heuristic", "frontier"]


class PairSpec(BaseModel):
    """One row of ``pairs.yaml``."""

    a: str
    b: str
    label: int
    messages: Path

    @field_validator("label")
    @classmethod
    def _label_is_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {v}")
        return v


def load_pairs_yaml(path: Path) -> list[PairSpec]:
    """Parse ``pairs.yaml`` into :class:`PairSpec` rows, ``messages`` resolved relative to it."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a YAML list of pair rows, got {type(raw).__name__}")
    specs: list[PairSpec] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {i} is not a mapping: {row!r}")
        spec = PairSpec.model_validate(row)
        if not spec.messages.is_absolute():
            spec = spec.model_copy(update={"messages": path.parent / spec.messages})
        specs.append(spec)
    return specs


def _is_bluebubbles_shaped(rows: list[dict[str, Any]]) -> bool:
    return any("ROWID" in r or "dateCreated" in r for r in rows)


def _rows_from_bluebubbles(payloads: list[dict[str, Any]], a: str, b: str) -> list[MessageRow]:
    """Adapt a raw BlueBubbles export to :class:`MessageRow`, mapping self/other onto a/b.

    A dyadic BlueBubbles export has exactly two senders (the export owner and the other
    party), so no address book is needed: ``isFromMe`` becomes ``a``, everyone else becomes
    ``b``.
    """
    guid_to_id = {
        str(p["guid"]): int(p["ROWID"])
        for p in payloads
        if p.get("guid") is not None and p.get("ROWID") is not None
    }
    self_sentinel = "\0self\0"
    group_id = f"{a}:{b}"
    rows: list[MessageRow] = []
    for payload in payloads:
        row = message_row_from_bluebubbles(
            payload, group_id, guid_to_id=guid_to_id, self_user_id=self_sentinel
        )
        if row is None:
            continue
        user_id = a if row.user_id == self_sentinel else b
        rows.append(row.model_copy(update={"user_id": user_id}))
    return rows


def _rows_from_generic(entries: list[dict[str, Any]], a: str, b: str) -> list[MessageRow]:
    """Adapt the generic ``{sender, text, sent_at_ms}`` shape to :class:`MessageRow`."""
    rows: list[MessageRow] = []
    for i, entry in enumerate(entries):
        try:
            sender = str(entry["sender"])
            text = str(entry["text"])
            sent_at_ms = int(entry["sent_at_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"generic message row {i} malformed: {entry!r}") from exc
        if sender not in (a, b):
            raise ValueError(f"generic message row {i}: sender {sender!r} is not {a!r} or {b!r}")
        rows.append(
            MessageRow(id=i, group_id=f"{a}:{b}", user_id=sender, sent_at_ms=sent_at_ms, body=text)
        )
    return rows


def load_pair_transcript(spec: PairSpec) -> Transcript:
    """Load ``spec.messages`` (either supported shape, auto-detected) into a real Transcript."""
    data = json.loads(spec.messages.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{spec.messages}: expected a non-empty JSON list of messages")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{spec.messages}: expected a JSON list of message objects")
    if _is_bluebubbles_shaped(data):
        rows = _rows_from_bluebubbles(data, spec.a, spec.b)
    else:
        rows = _rows_from_generic(data, spec.a, spec.b)
    if not rows:
        raise ValueError(f"{spec.messages}: no usable messages after parsing")
    return transcript_from_messages(rows, participants=[spec.a, spec.b])


def _extractor_config(name: LLMConfigName) -> ExtractorConfig:
    if name == "heuristic":
        return ExtractorConfig.heuristic()
    if name == "frontier":
        return ExtractorConfig.llm(llm_from_env("scoring"))
    raise ValueError(f"unknown --llm-config: {name!r}")  # pragma: no cover - argparse restricts


def features_for_pair(spec: PairSpec, config: ExtractorConfig) -> dict[str, Any]:
    """One §-matching known-pairs row: ``{a, b, label, features_ab, features_ba}``."""
    transcript = load_pair_transcript(spec)
    fab, fba = extract_features_sync(transcript, spec.a, spec.b, config)
    features_ab = dict(fab.features)
    features_ba = dict(fba.features)
    if transcript.source == "real" and transcript.has_timing():
        baselines = fit_baseline(
            {
                spec.a: response_latencies(transcript, spec.a, spec.b),
                spec.b: response_latencies(transcript, spec.b, spec.a),
            }
        )
        real_ab, real_ba = extract_real_features(transcript, spec.a, spec.b, baselines)
        features_ab.update(real_ab)
        features_ba.update(real_ba)
    return {
        "a": spec.a,
        "b": spec.b,
        "label": spec.label,
        "features_ab": features_ab,
        "features_ba": features_ba,
    }


def build_features(input_path: Path, output_path: Path, llm_config: LLMConfigName) -> int:
    """Run extraction over every pair in ``input_path`` and write ``output_path`` JSONL.

    Returns the number of pairs written.
    """
    specs = load_pairs_yaml(input_path)
    if not specs:
        raise ValueError(f"{input_path}: no pairs")
    config = _extractor_config(llm_config)
    with output_path.open("w") as out:
        for spec in specs:
            row = features_for_pair(spec, config)
            out.write(json.dumps(row) + "\n")
    return len(specs)


EXAMPLE_PAIRS_YAML = """\
# Known friend pairs (technical-architecture.md §5 build order, step 1).
# One row per pair; `label: 1` = known-good friends, `label: 0` = known-bad (e.g. acquaintances,
# a pair that drifted apart). Aim for 5-10 pairs you can already rank by hand.
- a: priya
  b: sam
  label: 1
  messages: priya_sam.example.json
- a: priya
  b: jordan
  label: 0
  messages: priya_jordan.example.json
"""

EXAMPLE_GENERIC_MESSAGES = [
    {
        "sender": "priya",
        "text": "omg you would NOT believe what happened today",
        "sent_at_ms": 1_700_000_000_000,
    },
    {
        "sender": "sam",
        "text": "what happened tell me everything",
        "sent_at_ms": 1_700_000_030_000,
    },
    {
        "sender": "priya",
        "text": "so you know how my landlord has been ghosting me",
        "sent_at_ms": 1_700_000_070_000,
    },
    {
        "sender": "sam",
        "text": "yes the heater thing right, did he finally answer",
        "sent_at_ms": 1_700_000_095_000,
    },
    {
        "sender": "priya",
        "text": "he showed up UNANNOUNCED at 8am",
        "sent_at_ms": 1_700_000_130_000,
    },
    {
        "sender": "sam",
        "text": "no way. are you okay? that's so invasive",
        "sent_at_ms": 1_700_000_150_000,
    },
    {
        "sender": "priya",
        "text": "i was fine just shaken, thank you for asking",
        "sent_at_ms": 1_700_000_180_000,
    },
    {
        "sender": "sam",
        "text": "of course. want me to come by after work?",
        "sent_at_ms": 1_700_000_210_000,
    },
]

EXAMPLE_GENERIC_MESSAGES_BAD = [
    {"sender": "priya", "text": "hey", "sent_at_ms": 1_700_000_000_000},
    {"sender": "jordan", "text": "hi", "sent_at_ms": 1_700_003_600_000},
    {"sender": "priya", "text": "how's it going", "sent_at_ms": 1_700_003_605_000},
    {"sender": "jordan", "text": "fine", "sent_at_ms": 1_700_007_200_000},
    {"sender": "priya", "text": "what did you do this weekend", "sent_at_ms": 1_700_007_205_000},
    {"sender": "jordan", "text": "nothing much", "sent_at_ms": 1_700_010_800_000},
    {"sender": "priya", "text": "did you watch the game", "sent_at_ms": 1_700_010_805_000},
    {"sender": "jordan", "text": "no", "sent_at_ms": 1_700_014_400_000},
]


def write_template(out_dir: Path, *, force: bool = False) -> list[Path]:
    """Write an example ``pairs.yaml`` plus example generic-JSON message exports.

    Refuses to overwrite existing files unless ``force`` is set. Returns the paths written.
    """
    pairs_path = out_dir / "pairs.yaml"
    good_path = out_dir / "priya_sam.example.json"
    bad_path = out_dir / "priya_jordan.example.json"
    for path in (pairs_path, good_path, bad_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(EXAMPLE_PAIRS_YAML)
    good_path.write_text(json.dumps(EXAMPLE_GENERIC_MESSAGES, indent=2) + "\n")
    bad_path.write_text(json.dumps(EXAMPLE_GENERIC_MESSAGES_BAD, indent=2) + "\n")
    return [pairs_path, good_path, bad_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.ops.known_pairs")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build", help="turn pairs.yaml + raw chat exports into a known-pairs features.jsonl"
    )
    build.add_argument("--input", type=Path, required=True, help="pairs.yaml")
    build.add_argument("--output", type=Path, required=True, help="features.jsonl to write")
    build.add_argument(
        "--llm-config",
        choices=("heuristic", "frontier"),
        default="heuristic",
        help="heuristic (default, free) or frontier (regulars.llm.factory.llm_from_env)",
    )

    template = sub.add_parser(
        "template", help="write an example pairs.yaml and message export to copy from"
    )
    template.add_argument(
        "--out-dir", type=Path, default=Path(), help="directory to write the example files into"
    )
    template.add_argument("--force", action="store_true", help="overwrite existing example files")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        n = build_features(args.input, args.output, args.llm_config)
        print(f"wrote {n} known pairs to {args.output}")
        return 0
    if args.command == "template":
        paths = write_template(args.out_dir, force=args.force)
        for path in paths:
            print(f"wrote {path}")
        return 0
    return 2  # pragma: no cover - argparse enforces the choice


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
