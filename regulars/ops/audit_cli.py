"""``python -m regulars.ops.audit_cli <subcommand>``

Operator-facing inspection of the audit log (§9: "every bot decision writes an audit row";
§10: "removals are explainable"). Today the only way to investigate a disputed removal, a
group that looks wrong, or a stuck payment is ad-hoc SQL against ``audit_log``. This gives
that investigation a stable, scriptable entry point.

Subcommands:

- ``user <user_id>``          every audit row touching this user, newest first.
- ``group <group_id>``        every audit row for a group, chronological (its story).
- ``why-removed <user_id> <group_id>``
                               the plain-English reason for a specific removal.
- ``stats``                    counts by ``decision_type``, a rough dashboard substitute.
- ``judge-versions``           distinct ``judge_version`` values with counts, from
                               ``predictions`` and ``audit_log``.

Reads ``DATABASE_URL`` from the environment by default (``regulars.db.engine.make_engine``);
pass ``--engine-url`` to point at a different database. Every subcommand accepts ``--json`` for
machine-readable output; without it, a plain-text table is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine

from regulars.core.audit import latest_removal
from regulars.db import repo
from regulars.db.engine import make_engine


def _open_engine(engine_url: str | None) -> Engine:
    return make_engine(engine_url)


def _parse_since(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; a naive value is treated as UTC (rows are stored UTC)."""
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _fmt_dt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _entry_json(entry: repo.AuditEntry) -> dict[str, Any]:
    return entry.model_dump(mode="json")


def render_entries(entries: Sequence[repo.AuditEntry]) -> str:
    """A readable table: when, decision_type, related group/plan ids, reason."""
    if not entries:
        return "(no audit rows)"
    header = f"{'WHEN':<20} {'DECISION_TYPE':<20} {'GROUP':<38} {'PLAN':<38} REASON"
    lines = [header]
    for e in entries:
        lines.append(
            f"{_fmt_dt(e.created_at):<20} {e.decision_type:<20} "
            f"{(e.group_id or '-'):<38} {(e.plan_id or '-'):<38} {e.reason}"
        )
    return "\n".join(lines)


# --- subcommands ---------------------------------------------------------------------------


def run_user(
    engine: Engine,
    user_id: str,
    *,
    limit: int | None = None,
    decision_type: str | None = None,
    since: datetime | None = None,
    as_json: bool = False,
) -> list[repo.AuditEntry]:
    """Every audit row touching ``user_id`` as ``subject_user_id``, newest first."""
    with engine.connect() as conn:
        entries = repo.audit_for_user(conn, user_id)
    entries = sorted(entries, key=lambda e: e.created_at, reverse=True)
    if decision_type is not None:
        entries = [e for e in entries if e.decision_type == decision_type]
    if since is not None:
        entries = [e for e in entries if e.created_at >= since]
    if limit is not None:
        entries = entries[:limit]
    if as_json:
        print(json.dumps([_entry_json(e) for e in entries], indent=2))
    else:
        print(render_entries(entries))
    return entries


def run_group(engine: Engine, group_id: str, *, as_json: bool = False) -> list[repo.AuditEntry]:
    """Every audit row for ``group_id``, chronological: the group's story."""
    with engine.connect() as conn:
        entries = repo.audit_for_group(conn, group_id)
    if as_json:
        print(json.dumps([_entry_json(e) for e in entries], indent=2))
    else:
        print(render_entries(entries))
    return entries


def run_why_removed(engine: Engine, user_id: str, group_id: str) -> str | None:
    """The plain-English reason ``user_id`` was removed from ``group_id``, or ``None``.

    Finds the same ``member_removed`` audit row that ``regulars.bot.router``'s "why was I
    removed" DM handler would, via the shared :func:`regulars.core.audit.latest_removal`
    helper, scoped to this specific group.
    """
    with engine.connect() as conn:
        entries = repo.audit_for_user(conn, user_id)
    latest = latest_removal(entries, group_id=group_id)
    if latest is None:
        print(f"No member_removed audit row for user {user_id} in group {group_id}.")
        return None
    print(f"{user_id} was removed from {group_id} on {_fmt_dt(latest.created_at)}: {latest.reason}")
    return latest.reason


def run_stats(
    engine: Engine,
    *,
    zip_code: str | None = None,
    since: datetime | None = None,
    as_json: bool = False,
) -> dict[str, int]:
    """Counts of audit rows by ``decision_type``."""
    with engine.connect() as conn:
        counts = repo.audit_decision_counts(conn, zip_code=zip_code, since=since)
    if as_json:
        print(json.dumps(counts, indent=2, sort_keys=True))
    elif not counts:
        print("(no audit rows)")
    else:
        width = max(len(k) for k in counts)
        for decision_type in sorted(counts):
            print(f"{decision_type:<{width}}  {counts[decision_type]}")
    return counts


def run_judge_versions(engine: Engine, *, as_json: bool = False) -> dict[str, dict[str, int]]:
    """Distinct ``judge_version`` values from ``predictions`` and ``audit_log``, with counts."""
    with engine.connect() as conn:
        counts = repo.judge_version_counts(conn)
    if as_json:
        print(json.dumps(counts, indent=2, sort_keys=True))
    elif not counts:
        print("(no judge_version values found)")
    else:
        print(f"{'JUDGE_VERSION':<30} {'PREDICTIONS':>11} {'AUDIT':>7}")
        for version in sorted(counts):
            c = counts[version]
            print(f"{version:<30} {c['predictions']:>11} {c['audit']:>7}")
    return counts


# --- argparse --------------------------------------------------------------------------------


def _add_engine_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-url", type=str, default=None, help="override DATABASE_URL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m regulars.ops.audit_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    user_p = sub.add_parser("user", help="every audit row touching this user, newest first")
    user_p.add_argument("user_id")
    user_p.add_argument("--limit", type=int, default=None)
    user_p.add_argument("--decision-type", type=str, default=None)
    user_p.add_argument("--since", type=str, default=None, help="ISO-8601 timestamp")
    user_p.add_argument("--json", action="store_true")
    _add_engine_url(user_p)

    group_p = sub.add_parser("group", help="every audit row for a group, chronologically")
    group_p.add_argument("group_id")
    group_p.add_argument("--json", action="store_true")
    _add_engine_url(group_p)

    why_p = sub.add_parser("why-removed", help="the plain-English reason a user was removed")
    why_p.add_argument("user_id")
    why_p.add_argument("group_id")
    _add_engine_url(why_p)

    stats_p = sub.add_parser("stats", help="counts by decision_type")
    stats_p.add_argument("--zip", dest="zip_code", type=str, default=None)
    stats_p.add_argument("--since", type=str, default=None, help="ISO-8601 timestamp")
    stats_p.add_argument("--json", action="store_true")
    _add_engine_url(stats_p)

    jv_p = sub.add_parser("judge-versions", help="distinct judge_version values, with counts")
    jv_p.add_argument("--json", action="store_true")
    _add_engine_url(jv_p)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = _open_engine(args.engine_url)
    try:
        if args.command == "user":
            since = _parse_since(args.since) if args.since else None
            run_user(
                engine,
                args.user_id,
                limit=args.limit,
                decision_type=args.decision_type,
                since=since,
                as_json=args.json,
            )
            return 0
        if args.command == "group":
            run_group(engine, args.group_id, as_json=args.json)
            return 0
        if args.command == "why-removed":
            reason = run_why_removed(engine, args.user_id, args.group_id)
            return 0 if reason is not None else 1
        if args.command == "stats":
            since = _parse_since(args.since) if args.since else None
            run_stats(engine, zip_code=args.zip_code, since=since, as_json=args.json)
            return 0
        if args.command == "judge-versions":
            run_judge_versions(engine, as_json=args.json)
            return 0
        return 2  # pragma: no cover - argparse enforces the choice
    finally:
        engine.dispose()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
