"""Group assembly: prefilter, max-min optimizer, exploration arm, prune/backfill, audit.

Pure functions and dataclasses (technical-architecture.md §6 Problem 1, §7, §8, §9). Nothing here
touches storage: every function returns what should be persisted, including the audit rows.
"""

from regulars.assembly.audit import DECISION_TYPES, AuditRecord, DecisionType
from regulars.assembly.exploration import (
    DEFAULT_EXPLORATION_RATE,
    AssemblyMethod,
    GroupPlan,
    RandomGroups,
    assemble_round,
    choose_method,
    leftover_users,
    random_groups,
)
from regulars.assembly.optimizer import (
    DEFAULT_CONTINUITY_BONUS,
    AssemblyResult,
    Continuity,
    PairScores,
    assemble_groups,
    group_score,
)
from regulars.assembly.prefilter import (
    Candidate,
    Pair,
    ZipAdjacency,
    is_plausible,
    pair_key,
    plausible_pairs,
    rejection_reason,
)
from regulars.assembly.rounds import (
    APPROVAL_THRESHOLD,
    MIN_RATERS,
    AttendanceRecord,
    DirectedRating,
    Removal,
    RoundUpdate,
    prune_and_backfill,
)

__all__ = [
    "APPROVAL_THRESHOLD",
    "DECISION_TYPES",
    "DEFAULT_CONTINUITY_BONUS",
    "DEFAULT_EXPLORATION_RATE",
    "MIN_RATERS",
    "AssemblyMethod",
    "AssemblyResult",
    "AttendanceRecord",
    "AuditRecord",
    "Candidate",
    "Continuity",
    "DecisionType",
    "DirectedRating",
    "GroupPlan",
    "Pair",
    "PairScores",
    "RandomGroups",
    "Removal",
    "RoundUpdate",
    "ZipAdjacency",
    "assemble_groups",
    "assemble_round",
    "choose_method",
    "group_score",
    "is_plausible",
    "leftover_users",
    "pair_key",
    "plausible_pairs",
    "prune_and_backfill",
    "random_groups",
    "rejection_reason",
]
