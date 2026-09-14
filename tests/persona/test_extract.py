import json

import pytest

from regulars.core import FakeLLM
from regulars.persona.extract import (
    DEFAULT_CONFIDENCE,
    PROSODY_CONFIDENCE,
    PersonaExtractionError,
    ProsodyHints,
    extract_persona,
    prosody_hints,
)
from regulars.persona.interview import InterviewAnswer
from regulars.persona.prosody import ProsodyFeatures

ANSWERS = [
    InterviewAnswer(prompt_index=0, text="Long hike, then cooked for six people."),
    InterviewAnswer(
        prompt_index=1, text="A friend who turned every story into his story. I just stopped."
    ),
    InterviewAnswer(prompt_index=2, text="People who actually show up.", typed=True),
]

FULL = {
    "conversational": {
        "opener_style": "a specific question about your week",
        "avg_turn_length": "medium",
        "humor_register": "dry",
        "question_tendency": "high",
        "interrupt_tendency": "waits, then lands one line",
        "silence_tolerance": "",
    },
    "content": {
        "unprompted_topics": ["cooking for people", "hiking"],
        "avoided_topics": [],
        "escalates_on": ["hosting"],
        "deflects_on": [],
    },
    "social": {
        "energy": "initiator",
        "disclosure_speed": "slow",
        "friction_points": ["someone turning every story into their story"],
        "drains": ["being the only one asking questions"],
    },
    "stated": {"exclusions": [], "seeking": "people who show up"},
    "confidence": {"conversational.humor_register": 0.9, "content.unprompted_topics": 0.8},
}


def _prosody(wpm: float, p90: float) -> ProsodyFeatures:
    return ProsodyFeatures(
        speech_rate_wpm=wpm,
        pause_count=3,
        pause_mean_ms=p90 * 0.7,
        pause_p90_ms=p90,
        laughter_count=0,
        duration_s=200,
    )


async def test_parses_valid_json_and_sets_provenance() -> None:
    llm = FakeLLM(replies=[json.dumps(FULL)])
    spec = await extract_persona(llm, ANSWERS)
    assert len(llm.calls) == 1
    assert spec.conversational.humor_register == "dry"
    assert spec.social.friction_points == ["someone turning every story into their story"]
    assert spec.has_negative_space()
    assert spec.provenance.sources == ["voice_interview"]
    conf = spec.provenance.confidence_by_field
    assert conf["conversational.humor_register"] == 0.9
    assert conf["content.unprompted_topics"] == 0.8
    assert conf["social.drains"] == DEFAULT_CONFIDENCE
    assert conf["conversational.avg_turn_length"] == DEFAULT_CONFIDENCE  # asserted default
    assert "conversational.silence_tolerance" not in conf  # empty, unrated
    assert "content.avoided_topics" not in conf

    system, messages = llm.calls[0]
    assert "friction_points" in system and "REQUIRED" in system
    assert "PROMPT 2 (voice)" in messages[0].content
    assert "PROMPT 3 (typed)" in messages[0].content
    assert ANSWERS[1].text in messages[0].content


async def test_tolerates_markdown_fences() -> None:
    llm = FakeLLM(replies=["Here you go:\n```json\n" + json.dumps(FULL) + "\n```"])
    spec = await extract_persona(llm, ANSWERS)
    assert spec.social.energy == "initiator"
    assert len(llm.calls) == 1


async def test_retries_once_on_invalid_json_with_error_appended() -> None:
    bad = json.dumps({**FULL, "social": {**FULL["social"], "energy": "loud"}})
    llm = FakeLLM(replies=[bad, json.dumps(FULL)])
    spec = await extract_persona(llm, ANSWERS)
    assert spec.social.energy == "initiator"
    assert len(llm.calls) == 2
    _, retry_messages = llm.calls[1]
    assert [m.role for m in retry_messages] == ["user", "assistant", "user"]
    assert retry_messages[1].content == bad
    assert "did not validate" in retry_messages[2].content
    assert "energy" in retry_messages[2].content  # pydantic's error names the field


async def test_fails_after_second_invalid_reply() -> None:
    llm = FakeLLM(replies=["nope", "still nope"])
    with pytest.raises(PersonaExtractionError):
        await extract_persona(llm, ANSWERS)
    assert len(llm.calls) == 2


async def test_reasks_when_negative_space_empty() -> None:
    hollow = json.loads(json.dumps(FULL))
    hollow["social"]["friction_points"] = []
    hollow["social"]["drains"] = []
    llm = FakeLLM(
        replies=[
            json.dumps(hollow),
            json.dumps({"friction_points": ["being talked over"], "drains": ["small talk"]}),
        ]
    )
    spec = await extract_persona(llm, ANSWERS)
    assert len(llm.calls) == 2
    _, reask = llm.calls[1]
    assert reask[-1].role == "user"
    assert "friction_points" in reask[-1].content
    assert reask[-2].role == "assistant"  # conversation continues from the valid draft
    assert spec.social.friction_points == ["being talked over"]
    assert spec.social.drains == ["small talk"]
    assert spec.has_negative_space()
    assert spec.provenance.confidence_by_field["social.drains"] == DEFAULT_CONFIDENCE


async def test_reask_keeps_existing_half_and_survives_garbage() -> None:
    half = json.loads(json.dumps(FULL))
    half["social"]["drains"] = []
    llm = FakeLLM(replies=[json.dumps(half), "I can't tell."])
    spec = await extract_persona(llm, ANSWERS)
    assert spec.social.friction_points == FULL["social"]["friction_points"]
    assert spec.social.drains == []
    assert spec.has_negative_space()  # friction alone counts
    assert len(llm.calls) == 2


def test_prosody_hint_mapping() -> None:
    assert prosody_hints(None) == ProsodyHints()
    assert prosody_hints(_prosody(140, 800)) == ProsodyHints()
    fast = prosody_hints(_prosody(190, 350))
    assert fast.avg_turn_length == "long"
    assert fast.silence_tolerance is not None and fast.silence_tolerance.startswith("low")
    slow = prosody_hints(_prosody(95, 400))
    assert slow.avg_turn_length == "short"
    long_pauses = prosody_hints(_prosody(140, 1500))
    assert long_pauses.avg_turn_length == "short"
    assert long_pauses.silence_tolerance is not None
    assert long_pauses.silence_tolerance.startswith("high")


async def test_prosody_overrides_spec_deterministically() -> None:
    llm = FakeLLM(replies=[json.dumps(FULL)])
    spec = await extract_persona(llm, ANSWERS, _prosody(190, 350))
    assert spec.conversational.avg_turn_length == "long"
    assert spec.conversational.silence_tolerance.startswith("low")
    conf = spec.provenance.confidence_by_field
    assert conf["conversational.avg_turn_length"] == PROSODY_CONFIDENCE
    assert conf["conversational.silence_tolerance"] == PROSODY_CONFIDENCE
    _, messages = llm.calls[0]
    assert "PROSODY" in messages[0].content and "190 wpm" in messages[0].content


async def test_per_answer_prosody_is_combined_when_none_passed() -> None:
    answers = [a.model_copy(update={"prosody": _prosody(190, 350)}) for a in ANSWERS]
    llm = FakeLLM(replies=[json.dumps(FULL)])
    spec = await extract_persona(llm, answers)
    assert spec.conversational.avg_turn_length == "long"


async def test_requires_answers() -> None:
    with pytest.raises(ValueError):
        await extract_persona(FakeLLM(), [])
