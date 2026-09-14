"""Operational scripts: not part of the runtime, run by hand or in CI/dev tooling.

``regulars.ops.seed`` builds a realistic-looking, internally-consistent demo database (users,
personas, sims, predictions, groups, a completed round with prune/backfill) so a person or another
agent has something browsable to click through without waiting on real voice interviews or dinners.
"""

from __future__ import annotations
