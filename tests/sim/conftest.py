from collections.abc import Sequence

import pytest

from regulars.core.llm import FakeLLM, LLMMessage
from regulars.core.persona import PersonaSpec


def echo_llm(model: str = "fake") -> FakeLLM:
    """Replies with a line that encodes how much history it saw, so turns are distinguishable."""

    def responder(system: str, messages: Sequence[LLMMessage]) -> str:
        return f"{model} line {len(messages)}"

    return FakeLLM(model=model, responder=responder)


@pytest.fixture
def spec_a() -> PersonaSpec:
    return PersonaSpec.model_validate(
        {
            "conversational": {"avg_turn_length": "short", "humor_register": "dry"},
            "content": {
                "unprompted_topics": ["bouldering", "bad landlords"],
                "escalates_on": ["people who say they 'don't really follow the news'"],
            },
            "social": {
                "energy": "initiator",
                "friction_points": ["being talked over", "small talk about the weather"],
                "drains": ["explaining my job"],
            },
            "stated": {"seeking": "people who argue well"},
        }
    )


@pytest.fixture
def spec_b() -> PersonaSpec:
    return PersonaSpec.model_validate(
        {
            "conversational": {"avg_turn_length": "long", "question_tendency": "high"},
            "content": {"unprompted_topics": ["the weather", "my dog"]},
            "social": {
                "energy": "responder",
                "friction_points": ["sarcasm at my expense"],
                "drains": ["politics"],
            },
        }
    )
