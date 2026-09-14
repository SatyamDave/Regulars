"""Transcript: the one shape both simulated and real conversations are reduced to.

Feature extraction (regulars.features) consumes only this type. Simulated transcripts have
``sent_at_ms=None`` on every turn; real chat transcripts carry wall-clock timestamps plus the
iMessage-only fields (reply_to, tapback_target). Real-only features must degrade to "absent",
not crash, when timestamps are missing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Source = Literal["sim", "real"]


class Turn(BaseModel):
    """One message or spoken turn."""

    speaker_id: str
    text: str
    sent_at_ms: int | None = None
    reply_to: int | None = Field(default=None, description="Index of the turn this replies to.")
    tapback_target: int | None = Field(
        default=None, description="Index of the turn this tapback reacts to; text holds the emoji."
    )

    @property
    def is_tapback(self) -> bool:
        return self.tapback_target is not None


class Transcript(BaseModel):
    """An ordered conversation among two or more participants."""

    participants: list[str]
    turns: list[Turn]
    source: Source
    metadata: dict[str, str | int | float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _speakers_are_participants(self) -> Transcript:
        known = set(self.participants)
        unknown = {t.speaker_id for t in self.turns} - known
        if unknown:
            raise ValueError(f"turns reference unknown speakers: {sorted(unknown)}")
        return self

    def turns_by(self, speaker_id: str) -> list[Turn]:
        return [t for t in self.turns if t.speaker_id == speaker_id and not t.is_tapback]

    def spoken(self) -> list[Turn]:
        """Turns that carry words (tapbacks excluded)."""
        return [t for t in self.turns if not t.is_tapback]

    def has_timing(self) -> bool:
        return all(t.sent_at_ms is not None for t in self.turns)

    def restrict(self, a: str, b: str) -> Transcript:
        """The dyadic sub-transcript between two participants of a group chat."""
        keep = {a, b}
        idx_map: dict[int, int] = {}
        turns: list[Turn] = []
        for i, t in enumerate(self.turns):
            if t.speaker_id not in keep:
                continue
            if t.is_tapback and t.tapback_target not in idx_map:
                continue  # reaction to someone outside the dyad
            reply_to = idx_map.get(t.reply_to) if t.reply_to is not None else None
            target = idx_map[t.tapback_target] if t.tapback_target is not None else None
            idx_map[i] = len(turns)
            turns.append(t.model_copy(update={"reply_to": reply_to, "tapback_target": target}))
        return Transcript(
            participants=[a, b], turns=turns, source=self.source, metadata=self.metadata
        )
