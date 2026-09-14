"""Operator-facing tooling: the glue between raw/synthetic data and the running system.

Nothing here is imported by the running system (bot, judge, sim); these are one-off/founder-run
scripts. ``regulars.ops.audit_cli`` inspects the audit log (§9, §10). ``regulars.ops.known_pairs``
turns 5-10 known friend pairs' real chat exports into the ``features.jsonl`` that
``regulars.judge.cli known-pairs`` consumes (§5 build-order step 1). ``regulars.ops.benchmark_cli``
load-tests the assembly optimizer (§7, §8). ``regulars.ops.seed`` builds a realistic-looking,
internally-consistent demo database (users, personas, sims, predictions, groups, a completed round
with prune/backfill) so a person or another agent has something browsable to click through without
waiting on real voice interviews or dinners.
"""

from __future__ import annotations
