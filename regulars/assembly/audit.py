"""Audit records for assembly decisions (technical-architecture.md §9, §10).

Every function in ``regulars.assembly`` is pure and returns, alongside its result, the
``AuditRecord`` rows that the caller must persist. The field names line up one-to-one with
``regulars.db.repo.write_audit`` keyword arguments (``decision_type``, ``subject_user_id``,
``reason``, ``payload``) so persisting is a straight pass-through; ``group_members`` is carried
here because the group row does not exist yet when the decision is made, and the caller adds the
``group_id`` once it does.

``reason`` is plain English fit to show the affected user (§10: removals are explainable).
``payload`` is JSON-serialisable: non-finite scores are stored as ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

DecisionType = Literal["group_assembled", "member_removed", "member_added", "method_chosen"]
DECISION_TYPES: frozenset[str] = frozenset(get_args(DecisionType))


@dataclass(frozen=True)
class AuditRecord:
    decision_type: DecisionType
    subject_user_id: str | None
    group_members: tuple[str, ...]
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"decision_type must be one of {sorted(DECISION_TYPES)}")
        if not self.reason.strip():
            raise ValueError("audit reason must not be empty")


def json_score(value: float) -> float | None:
    """Scores go into ``payload``; ``-inf`` (unscored pair) is not JSON, so it becomes ``None``."""
    return value if math.isfinite(value) else None


def members_key(members: Iterable[str]) -> tuple[str, ...]:
    """Canonical (sorted) member tuple used everywhere a group is recorded."""
    return tuple(sorted(members))
