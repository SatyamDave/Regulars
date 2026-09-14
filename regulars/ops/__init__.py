"""Founder-facing operational scripts: the glue between raw exports and the pipeline.

Nothing here is imported by the running system (bot, judge, sim); this package exists for
one-off runs against real data, starting with technical-architecture.md §5's build-order
step 1: turn 5-10 known friend pairs' real chat exports into the ``features.jsonl`` that
``regulars.judge.cli known-pairs`` consumes.
"""
