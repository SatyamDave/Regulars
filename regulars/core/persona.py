"""PersonaSpec: a structured behavioral spec. Text fields, never embeddings.

Personas are immutable once stored; a change is a new version (see regulars.persona). The
``friction_points`` and ``drains`` fields are load-bearing for the simulator: without explicit
negative space two LLM personas are pleasant to each other and the sim has no variance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PersonaSource = Literal["voice_interview", "chat_behavior", "chatlog_import", "user_correction"]


class Conversational(BaseModel):
    opener_style: str = ""
    avg_turn_length: Literal["short", "medium", "long"] = "medium"
    humor_register: Literal["dry", "absurdist", "self-deprecating", "none"] = "none"
    question_tendency: Literal["high", "low"] = "low"
    interrupt_tendency: str = ""
    silence_tolerance: str = ""


class Content(BaseModel):
    unprompted_topics: list[str] = Field(default_factory=list)
    avoided_topics: list[str] = Field(default_factory=list)
    escalates_on: list[str] = Field(default_factory=list)
    deflects_on: list[str] = Field(default_factory=list)


class Social(BaseModel):
    energy: Literal["initiator", "responder", "observer"] = "responder"
    disclosure_speed: Literal["fast", "slow"] = "slow"
    friction_points: list[str] = Field(default_factory=list)
    drains: list[str] = Field(default_factory=list)


class Stated(BaseModel):
    exclusions: list[str] = Field(default_factory=list)
    seeking: str = ""


class Provenance(BaseModel):
    sources: list[PersonaSource] = Field(default_factory=list)
    confidence_by_field: dict[str, float] = Field(default_factory=dict)


class PersonaSpec(BaseModel):
    conversational: Conversational = Field(default_factory=Conversational)
    content: Content = Field(default_factory=Content)
    social: Social = Field(default_factory=Social)
    stated: Stated = Field(default_factory=Stated)
    provenance: Provenance = Field(default_factory=Provenance)

    def has_negative_space(self) -> bool:
        """True when the spec carries the fields the simulator needs to produce variance."""
        return bool(self.social.friction_points or self.social.drains)
