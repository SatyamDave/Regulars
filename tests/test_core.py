import pytest

from regulars.core import FEATURE_NAMES, FakeLLM, LLMClient, PersonaSpec, Transcript, Turn


def test_transcript_rejects_unknown_speaker() -> None:
    with pytest.raises(ValueError):
        Transcript(participants=["a"], turns=[Turn(speaker_id="b", text="hi")], source="sim")


def test_restrict_remaps_reply_indices() -> None:
    t = Transcript(
        participants=["a", "b", "c"],
        turns=[
            Turn(speaker_id="a", text="hi"),
            Turn(speaker_id="c", text="yo"),
            Turn(speaker_id="b", text="hey", reply_to=0),
            Turn(speaker_id="b", text="❤️", tapback_target=1),
        ],
        source="real",
    )
    d = t.restrict("a", "b")
    assert [x.text for x in d.turns] == ["hi", "hey"]
    assert d.turns[1].reply_to == 0


def test_persona_negative_space() -> None:
    assert not PersonaSpec().has_negative_space()
    p = PersonaSpec.model_validate({"social": {"friction_points": ["being talked over"]}})
    assert p.has_negative_space()


async def test_fake_llm_is_a_client() -> None:
    fake = FakeLLM(replies=["ok"])
    assert isinstance(fake, LLMClient)
    assert await fake.complete("sys", []) == "ok"


def test_feature_registry_unique() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
