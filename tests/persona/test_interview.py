import pytest

from regulars.persona.interview import INTERVIEW_PROMPTS, InterviewAnswer, InterviewState
from regulars.persona.prosody import ProsodyFeatures


def test_prompts_match_product_plan() -> None:
    assert INTERVIEW_PROMPTS == (
        "Walk me through your last genuinely good weekend.",
        "Tell me about someone you stopped hanging out with, and why.",
        "What are you actually looking for right now?",
    )


def test_state_machine_walks_prompts_in_order() -> None:
    s0 = InterviewState()
    assert s0.next_prompt_index == 0
    assert s0.next_prompt == INTERVIEW_PROMPTS[0]
    assert not s0.is_complete

    s1 = s0.record_answer("Hiked, then a long dinner.")
    assert s0.answers == ()  # immutable: original untouched
    assert s1.next_prompt_index == 1
    assert s1.answers[0] == InterviewAnswer(prompt_index=0, text="Hiked, then a long dinner.")

    prosody = ProsodyFeatures(
        speech_rate_wpm=150,
        pause_count=2,
        pause_mean_ms=400,
        pause_p90_ms=600,
        laughter_count=1,
        duration_s=80,
    )
    s2 = s1.record_answer("A guy who only talked about himself.", prosody)
    assert s2.answers[1].prosody == prosody
    assert s2.answers[1].prompt == INTERVIEW_PROMPTS[1]
    assert not s2.is_complete

    s3 = s2.record_answer("people who show up", typed=True)
    assert s3.is_complete
    assert s3.next_prompt is None
    assert s3.answers[2].typed is True

    with pytest.raises(ValueError, match="already complete"):
        s3.record_answer("one more")


def test_blank_answer_rejected() -> None:
    with pytest.raises(ValueError):
        InterviewState().record_answer("   ")


def test_answers_must_be_contiguous() -> None:
    with pytest.raises(ValueError):
        InterviewState(answers=(InterviewAnswer(prompt_index=1, text="x"),))


def test_state_round_trips_through_json() -> None:
    s = InterviewState().record_answer("a").record_answer("b")
    again = InterviewState.model_validate_json(s.model_dump_json())
    assert again == s
    assert again.next_prompt_index == 2
