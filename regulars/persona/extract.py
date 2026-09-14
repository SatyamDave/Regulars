"""Voice interview → :class:`PersonaSpec` (§2 Extraction).

The LLM is asked for strict JSON matching the four content sections of the spec. Parsing is
pydantic; a parse failure is retried exactly once with the validation error appended to the
conversation. ``friction_points`` and ``drains`` are load-bearing (§2): if the model leaves them
empty we re-ask for them specifically, once.

Prosody mapping (deterministic; prosody wins over the transcript for pacing fields because a
text transcript destroys exactly this signal, §2):

==================================================  ===================  =====================
condition                                           ``avg_turn_length``  ``silence_tolerance``
==================================================  ===================  =====================
rate >= ``FAST_WPM`` and p90 pause < ``SHORT_PAUSE_MS``  ``"long"``      low (fills silence)
rate <= ``SLOW_WPM`` or p90 pause >= ``LONG_PAUSE_MS``   ``"short"``     high (sits with it)
anything else                                       LLM value kept       LLM value kept
==================================================  ===================  =====================

Fields overridden this way get ``confidence_by_field[path] = PROSODY_CONFIDENCE``. The prosody
summary is also shown to the model so it can inform the fields we do not override
(``interrupt_tendency``, ``humor_register``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from regulars.core.llm import LLMClient, LLMMessage
from regulars.core.persona import (
    Content,
    Conversational,
    PersonaSource,
    PersonaSpec,
    Provenance,
    Social,
    Stated,
)
from regulars.persona._fields import default_value, field_paths, get_value
from regulars.persona.interview import InterviewAnswer
from regulars.persona.prosody import ProsodyFeatures, combine

FAST_WPM = 170.0
SLOW_WPM = 110.0
SHORT_PAUSE_MS = 500.0
LONG_PAUSE_MS = 1200.0

DEFAULT_CONFIDENCE = 0.6
"""Confidence for a populated field when the model did not rate it."""
PROSODY_CONFIDENCE = 0.7
EXTRACTION_TEMPERATURE = 0.2
EXTRACTION_MAX_TOKENS = 2048


class PersonaExtractionError(RuntimeError):
    """The model failed to produce valid JSON twice in a row."""


class PersonaDraft(BaseModel):
    """What the LLM returns: the four content sections plus optional self-rated confidence."""

    conversational: Conversational = Field(default_factory=Conversational)
    content: Content = Field(default_factory=Content)
    social: Social = Field(default_factory=Social)
    stated: Stated = Field(default_factory=Stated)
    confidence: dict[str, float] = Field(default_factory=dict)


class NegativeSpace(BaseModel):
    friction_points: list[str] = Field(default_factory=list)
    drains: list[str] = Field(default_factory=list)


_FIELD_GUIDE = """\
Field meanings:
- conversational.opener_style: how they start a conversation, in their own register.
- conversational.avg_turn_length: short | medium | long.
- conversational.humor_register: dry | absurdist | self-deprecating | none.
- conversational.question_tendency: high | low — do they ask or tell?
- conversational.interrupt_tendency: free text; do they jump in, wait, or trail off?
- conversational.silence_tolerance: free text; how they handle a lull.
- content.unprompted_topics: what they raised without being asked.
- content.avoided_topics: what they skipped or brushed past when the prompt invited it.
- content.escalates_on: topics where their energy rose.
- content.deflects_on: topics they redirected away from.
- social.energy: initiator | responder | observer.
- social.disclosure_speed: fast | slow — how quickly they got personal.
- social.friction_points: REQUIRED, non-empty. Concrete behaviors in other people that grate on
  them. Read the second answer closely; it is about someone they dropped.
- social.drains: REQUIRED, non-empty. Situations or dynamics that leave them tired.
- stated.exclusions: things they explicitly said they do not want. Quote, do not infer.
- stated.seeking: what they said they are looking for, in one sentence.
- confidence: map of "section.field" -> 0.0-1.0, your confidence in each populated field.

Rules:
- Reply with ONE JSON object and nothing else: no prose, no markdown fences.
- Use only the enum values listed. Use [] or "" when there is truly no evidence, EXCEPT
  social.friction_points and social.drains, which must each have at least one entry.
- Be specific and idiosyncratic. "Doesn't like rude people" is useless; "checks out when someone
  talks over the quietest person at the table" is what we need.
- Never invent facts about the person's identity, job, or location. Behavior only.
"""


def system_prompt(task: str) -> str:
    """Stable system prompt: role, task, field guide, JSON schema (in that order, for caching)."""
    schema = json.dumps(PersonaDraft.model_json_schema(), separators=(",", ":"))
    return (
        "You build behavioral persona specs for a dinner-matching service. The spec drives a "
        "simulation of how this person talks in a group chat, so behavior beats biography.\n\n"
        f"{task}\n\n{_FIELD_GUIDE}\nJSON schema for your reply:\n{schema}"
    )


INTERVIEW_TASK = (
    "You will read a three-prompt voice interview (transcribed) and optional prosody features "
    "measured from the audio. Produce the persona spec."
)

RETRY_TEMPLATE = (
    "That reply did not validate against the schema. Error:\n{error}\n\n"
    "Reply again with ONLY the corrected JSON object."
)

NEGATIVE_SPACE_REASK = (
    "Your spec left social.friction_points and/or social.drains empty. These are required: "
    "without them the simulation has nothing to push against. Re-read the interview, especially "
    "the answer about someone they stopped hanging out with, and reply with ONLY this JSON: "
    '{"friction_points": ["..."], "drains": ["..."]} — each list non-empty, each entry a '
    "concrete behavior or dynamic."
)


def prosody_summary(prosody: ProsodyFeatures) -> str:
    pitch = (
        f"{prosody.pitch_variance:.0f} Hz^2" if prosody.pitch_variance is not None else "unmeasured"
    )
    return (
        f"speech rate {prosody.speech_rate_wpm:.0f} wpm; {prosody.pause_count} pauses "
        f"(mean {prosody.pause_mean_ms:.0f} ms, p90 {prosody.pause_p90_ms:.0f} ms); "
        f"pitch variance {pitch}; laughter x{prosody.laughter_count}; "
        f"{prosody.duration_s:.0f}s of audio"
    )


def format_answers(answers: Sequence[InterviewAnswer]) -> str:
    blocks = []
    for a in answers:
        mode = "typed" if a.typed else "voice"
        blocks.append(f"PROMPT {a.prompt_index + 1} ({mode}): {a.prompt}\nANSWER: {a.text}")
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class ProsodyHints:
    avg_turn_length: str | None = None
    silence_tolerance: str | None = None

    def as_overrides(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.avg_turn_length is not None:
            out["conversational.avg_turn_length"] = self.avg_turn_length
        if self.silence_tolerance is not None:
            out["conversational.silence_tolerance"] = self.silence_tolerance
        return out


def prosody_hints(prosody: ProsodyFeatures | None) -> ProsodyHints:
    """The deterministic mapping documented in the module docstring."""
    if prosody is None or prosody.duration_s <= 0:
        return ProsodyHints()
    if prosody.speech_rate_wpm >= FAST_WPM and prosody.pause_p90_ms < SHORT_PAUSE_MS:
        return ProsodyHints(
            avg_turn_length="long",
            silence_tolerance="low: fills silence quickly (fast speech, short pauses)",
        )
    if prosody.speech_rate_wpm <= SLOW_WPM or prosody.pause_p90_ms >= LONG_PAUSE_MS:
        return ProsodyHints(
            avg_turn_length="short",
            silence_tolerance="high: comfortable sitting with a pause (measured pace)",
        )
    return ProsodyHints()


def _strip_to_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object in reply", text, 0)
    return text[start : end + 1]


def parse_model[T: BaseModel](text: str, model_type: type[T]) -> T:
    """Lenient-on-wrapping, strict-on-content: strips fences/prose, then validates."""
    data: Any = json.loads(_strip_to_json(text))
    return model_type.model_validate(data)


async def request_json[T: BaseModel](
    llm: LLMClient,
    system: str,
    messages: Sequence[LLMMessage],
    model_type: type[T],
    *,
    max_tokens: int = EXTRACTION_MAX_TOKENS,
) -> tuple[T, list[LLMMessage]]:
    """Ask for JSON; on a parse failure retry once with the error appended.

    Returns the parsed model and the full message history (including the assistant's valid
    reply) so callers can continue the conversation.
    """
    history = list(messages)
    raw = await llm.complete(
        system, history, temperature=EXTRACTION_TEMPERATURE, max_tokens=max_tokens
    )
    try:
        parsed = parse_model(raw, model_type)
    except (json.JSONDecodeError, ValidationError) as first:
        history += [
            LLMMessage("assistant", raw),
            LLMMessage("user", RETRY_TEMPLATE.format(error=first)),
        ]
        raw = await llm.complete(
            system, history, temperature=EXTRACTION_TEMPERATURE, max_tokens=max_tokens
        )
        try:
            parsed = parse_model(raw, model_type)
        except (json.JSONDecodeError, ValidationError) as second:
            raise PersonaExtractionError(
                f"model returned invalid {model_type.__name__} JSON twice: {second}"
            ) from second
    return parsed, [*history, LLMMessage("assistant", raw)]


async def ensure_negative_space(
    llm: LLMClient, system: str, history: Sequence[LLMMessage], draft: PersonaDraft
) -> PersonaDraft:
    """Re-ask once for friction_points/drains if either is empty. Never raises on failure."""
    if draft.social.friction_points and draft.social.drains:
        return draft
    raw = await llm.complete(
        system,
        [*history, LLMMessage("user", NEGATIVE_SPACE_REASK)],
        temperature=EXTRACTION_TEMPERATURE,
        max_tokens=512,
    )
    try:
        extra = parse_model(raw, NegativeSpace)
    except (json.JSONDecodeError, ValidationError):
        return draft
    social = draft.social.model_copy(
        update={
            "friction_points": draft.social.friction_points or extra.friction_points,
            "drains": draft.social.drains or extra.drains,
        }
    )
    return draft.model_copy(update={"social": social})


def draft_to_spec(
    draft: PersonaDraft,
    source: PersonaSource,
    *,
    overrides: dict[str, str] | None = None,
    default_confidence: float = DEFAULT_CONFIDENCE,
) -> PersonaSpec:
    """Assemble a spec: apply deterministic overrides, then fill confidence for populated fields."""
    data = draft.model_dump(exclude={"confidence"})
    for path, value in (overrides or {}).items():
        section, field = path.split(".", 1)
        data[section][field] = value
    spec = PersonaSpec.model_validate(data)

    # A confidence entry marks a field as known even when its value equals the default (a
    # Literal the model asserted explicitly); see _fields.is_populated. Empty strings/lists the
    # model returned mean "no evidence" and get no entry.
    confidence: dict[str, float] = {}
    for path in field_paths():
        section, field = path.split(".", 1)
        value = get_value(spec, path)
        asserted = field in getattr(draft, section).model_fields_set and value not in ("", [])
        if path in (overrides or {}):
            confidence[path] = PROSODY_CONFIDENCE
        elif asserted or value != default_value(path) or path in draft.confidence:
            rated = draft.confidence.get(path, default_confidence)
            confidence[path] = min(1.0, max(0.0, rated))
    spec.provenance = Provenance(sources=[source], confidence_by_field=confidence)
    return spec


async def extract_persona(
    llm: LLMClient,
    answers: Sequence[InterviewAnswer],
    prosody: ProsodyFeatures | None = None,
) -> PersonaSpec:
    """Interview answers (+ optional prosody) → spec with ``sources=["voice_interview"]``.

    ``prosody`` overrides any per-answer prosody attached to ``answers``; when it is ``None`` the
    per-answer features are combined duration-weighted.
    """
    if not answers:
        raise ValueError("extract_persona needs at least one interview answer")
    merged = prosody or combine([a.prosody for a in answers if a.prosody is not None])
    system = system_prompt(INTERVIEW_TASK)
    user = format_answers(answers)
    if merged is not None:
        user += f"\n\nPROSODY (measured from audio): {prosody_summary(merged)}"
    draft, history = await request_json(llm, system, [LLMMessage("user", user)], PersonaDraft)
    draft = await ensure_negative_space(llm, system, history, draft)
    return draft_to_spec(draft, "voice_interview", overrides=prosody_hints(merged).as_overrides())
