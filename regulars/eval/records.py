"""Flat evaluation records and their JSONL loader.

These are *denormalized* rows: the joins the §1 schema needs (rating × prediction × group,
membership × group, attendance × plan) are done upstream so this package can stay free of the
database and the judge. Every record is immutable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

AssemblyMethod = Literal["model", "random"]
"""§6 Problem 1: ``random`` is the exploration arm and the only unbiased evaluation set."""

COMMITTED_RSVP_STATES: frozenset[str] = frozenset({"yes", "confirmed", "paid", "locked"})


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabeledDyad(_Record):
    """One DIRECTED rating (rater → ratee) joined to the prediction made for that pair.

    ``y_hat`` is the raw judge score, ``y_hat_calibrated`` the isotonic-calibrated
    ``P(would_see_again)`` when available (§5 Calibration). a→b and b→a are separate rows.
    """

    group_id: str
    plan_id: str
    rater: str
    ratee: str
    y_hat: float
    y_hat_calibrated: float | None = None
    would_see_again: bool
    felt_connected_1_7: int | None = Field(default=None, ge=1, le=7)
    assembly_method: AssemblyMethod
    round_index: int = Field(ge=1)

    @property
    def outcome(self) -> float:
        """Real rating on a single scale: ``felt_connected_1_7`` if present, else 0/1."""
        if self.felt_connected_1_7 is not None:
            return float(self.felt_connected_1_7)
        return float(self.would_see_again)


class Membership(_Record):
    """``memberships`` joined to ``groups.assembly_method``.

    A user is a member in round ``r`` iff ``joined_round <= r`` and
    (``left_round`` is None or ``left_round > r``).
    """

    group_id: str
    user_id: str
    joined_round: int = Field(ge=1)
    left_round: int | None = None
    exit_reason: str | None = None
    assembly_method: AssemblyMethod

    def member_in_round(self, round_index: int) -> bool:
        return self.joined_round <= round_index and (
            self.left_round is None or self.left_round > round_index
        )

    @property
    def active(self) -> bool:
        return self.left_round is None


class Attendance(_Record):
    """``attendance`` joined to ``plans.group_id`` and the plan's round within its group."""

    plan_id: str
    group_id: str
    user_id: str
    round_index: int = Field(ge=1)
    attended: bool


class Rsvp(_Record):
    """One RSVP. ``committed`` means the user said yes or, where deposits run, paid."""

    plan_id: str
    user_id: str
    state: str
    paid: bool = False

    committed_states: ClassVar[frozenset[str]] = COMMITTED_RSVP_STATES

    @property
    def committed(self) -> bool:
        return self.paid or self.state.lower() in self.committed_states


def iter_jsonl[R: _Record](path: str | Path, model: type[R]) -> Iterator[R]:
    """Yield one validated record per non-blank line."""
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield model.model_validate(json.loads(stripped))
            except ValueError as exc:  # json.JSONDecodeError and pydantic.ValidationError
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_jsonl[R: _Record](path: str | Path, model: type[R]) -> list[R]:
    return list(iter_jsonl(path, model))


def load_dyads(path: str | Path) -> list[LabeledDyad]:
    return load_jsonl(path, LabeledDyad)


def dump_jsonl(path: str | Path, records: Iterable[_Record]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json())
            fh.write("\n")
