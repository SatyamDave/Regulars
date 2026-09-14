"""Post-dinner prune and backfill (technical-architecture.md §7 continuity, §9, §10, §11 step 3).

After each dinner a group's composition may change. Two rules remove a member; both are
mechanical, both are recorded with a reason written for the person being removed (§10:
removals are explainable), and nothing else removes anyone.

**Removal rules**

1. *No-show after paying.* The member paid the deposit and did not attend. Members who never
   paid (no commitment, no seat lost) are not removed by this rule.
2. *Low rating.* Among the **other** members who attended and rated this member, at least
   ``MIN_RATERS`` (3) did so, and fewer than ``APPROVAL_THRESHOLD`` (50%) said they would see
   them again. With fewer than three raters there is not enough signal and nobody is removed.
   Ratings are directed (§1): only ratings *about* the member count, never their own ratings of
   others. Ratings from non-members, from members who did not attend, and self-ratings are
   ignored.

Removals are applied in severity order (no-shows first, then lowest approval share). A removal
that would leave fewer than ``min_keep`` members is *not* applied: the group is kept viable for
backfilling around, and the spared member is listed in ``RoundUpdate.notes``.

**Backfill** refills the group to its previous size from ``queue`` (waiting users, in priority
order), each time adding the candidate that leaves the group's minimum pair score highest; ties
go to the earlier queue position. A candidate with no score against a current member is never
added (an unscored pair never co-groups). Backfill uses the raw judge scores; the continuity
bonus is an assembly concern (:mod:`regulars.assembly.optimizer`) and does not apply here since
the surviving members are, by construction, the ones kept together.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from regulars.assembly.audit import AuditRecord, json_score, members_key
from regulars.assembly.optimizer import PairScores, group_score

MIN_RATERS = 3
APPROVAL_THRESHOLD = 0.5


@dataclass(frozen=True)
class DirectedRating:
    """``rater`` says whether they would see ``ratee`` again. a->b is not b->a (§1)."""

    rater: str
    ratee: str
    would_see_again: bool


@dataclass(frozen=True)
class AttendanceRecord:
    user_id: str
    attended: bool
    paid: bool = False


class Removal(NamedTuple):
    user_id: str
    reason: str


@dataclass(frozen=True)
class RoundUpdate:
    keep: tuple[str, ...]
    """Members retained from the previous round, before backfill."""
    removed: list[Removal]
    added: tuple[str, ...]
    audit: list[AuditRecord]
    min_score: float
    """Raw minimum pair score of ``keep + added``; ``inf`` for fewer than two members."""
    notes: tuple[str, ...] = ()

    @property
    def members(self) -> tuple[str, ...]:
        return members_key((*self.keep, *self.added))


@dataclass(frozen=True)
class _Verdict:
    user_id: str
    rule: str
    severity: tuple[int, float]
    reason: str
    payload: dict[str, object]


def _no_show_reason() -> str:
    return (
        "You put down a deposit for this dinner but didn't come. Tables only work when everyone "
        "who commits shows up, so we've opened your seat to someone else."
    )


def _low_rating_reason(raters: int, approvals: int) -> str:
    return (
        f"After the dinner, {approvals} of the {raters} people who rated you said they'd want to "
        "see you again, which is fewer than half. We're moving you to a different table where "
        "the fit should be better."
    )


def _verdicts(
    group: Sequence[str],
    ratings: Iterable[DirectedRating],
    attendance: Iterable[AttendanceRecord],
) -> list[_Verdict]:
    members = set(group)
    by_user = {a.user_id: a for a in attendance if a.user_id in members}
    attended = {u for u, a in by_user.items() if a.attended}

    approvals: dict[str, int] = dict.fromkeys(group, 0)
    raters: dict[str, int] = dict.fromkeys(group, 0)
    for r in ratings:
        if r.ratee not in members or r.rater not in attended or r.rater == r.ratee:
            continue
        raters[r.ratee] += 1
        approvals[r.ratee] += int(r.would_see_again)

    out: list[_Verdict] = []
    for u in group:
        att = by_user.get(u)
        if att is not None and att.paid and not att.attended:
            out.append(
                _Verdict(
                    user_id=u,
                    rule="no_show",
                    severity=(0, 0.0),
                    reason=_no_show_reason(),
                    payload={"rule": "no_show", "paid": True, "attended": False},
                )
            )
            continue
        n = raters[u]
        if n >= MIN_RATERS and approvals[u] / n < APPROVAL_THRESHOLD:
            share = approvals[u] / n
            out.append(
                _Verdict(
                    user_id=u,
                    rule="low_rating",
                    severity=(1, share),
                    reason=_low_rating_reason(n, approvals[u]),
                    payload={
                        "rule": "low_rating",
                        "raters": n,
                        "approvals": approvals[u],
                        "approval_share": share,
                        "min_raters": MIN_RATERS,
                        "threshold": APPROVAL_THRESHOLD,
                    },
                )
            )
    out.sort(key=lambda v: (v.severity, v.user_id))
    return out


def prune_and_backfill(
    group: Sequence[str],
    ratings: Iterable[DirectedRating],
    attendance: Iterable[AttendanceRecord],
    queue: Sequence[str],
    scores: PairScores,
    min_keep: int = 3,
    *,
    group_size: int | None = None,
) -> RoundUpdate:
    """Apply the removal rules to ``group`` and refill from ``queue`` maximising the new minimum.

    ``group_size`` defaults to the group's current size. See the module docstring for the rules.
    """
    if len(set(group)) != len(group):
        raise ValueError("group has duplicate members")
    if min_keep < 0:
        raise ValueError("min_keep must be non-negative")
    target = len(group) if group_size is None else group_size
    original = members_key(group)

    keep: list[str] = list(group)
    removed: list[Removal] = []
    audit: list[AuditRecord] = []
    notes: list[str] = []
    for v in _verdicts(group, ratings, attendance):
        if len(keep) - 1 < min_keep:
            notes.append(
                f"{v.user_id} met the {v.rule} removal rule but was kept: removing them would "
                f"leave fewer than {min_keep} members"
            )
            continue
        keep.remove(v.user_id)
        removed.append(Removal(v.user_id, v.reason))
        audit.append(
            AuditRecord(
                decision_type="member_removed",
                subject_user_id=v.user_id,
                group_members=original,
                reason=v.reason,
                payload=dict(v.payload),
            )
        )

    excluded = set(group)
    candidates = [u for u in queue if u not in excluded]
    added: list[str] = []
    current = list(keep)
    while len(current) < target and candidates:
        best_i: int | None = None
        best_min = -math.inf
        for i, c in enumerate(candidates):
            m = group_score([*current, c], scores)
            if m == -math.inf:
                continue
            if best_i is None or m > best_min:
                best_i, best_min = i, m
        if best_i is None:
            notes.append("backfill stopped: no waiting user is scored against every member")
            break
        u = candidates.pop(best_i)
        current.append(u)
        added.append(u)
        audit.append(
            AuditRecord(
                decision_type="member_added",
                subject_user_id=u,
                group_members=members_key(current),
                reason=(
                    "You've been added to fill an open seat: of everyone waiting, you had the "
                    "strongest predicted fit with the people already at this table."
                ),
                payload={
                    "resulting_min": json_score(best_min),
                    "queue_position": list(queue).index(u),
                    "seat_of": [r.user_id for r in removed],
                },
            )
        )

    return RoundUpdate(
        keep=tuple(keep),
        removed=removed,
        added=tuple(added),
        audit=audit,
        min_score=group_score(current, scores),
        notes=tuple(notes),
    )
