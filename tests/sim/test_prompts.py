import pytest

from regulars.core.persona import PersonaSpec
from regulars.sim.prompts import BANNED_PHRASES, enactment_system_prompt, render_spec
from regulars.sim.scenarios import SCENARIOS

# Extra literal phrases beyond the module list: the test must not depend solely on a list the
# module itself controls.
EXTRA_BANNED = (
    "have a good conversation",
    "build rapport",
    "be a good listener",
    "make the other person feel",
    "be supportive",
)


def test_render_spec_contains_negative_space(spec_a: PersonaSpec) -> None:
    text = render_spec(spec_a)
    assert "being talked over" in text
    assert "small talk about the weather" in text
    assert "explaining my job" in text
    assert "don't really follow the news" in text
    assert "people who argue well" in text
    assert "confidence_by_field" not in text  # provenance omitted


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
@pytest.mark.parametrize("inject", [False, True])
def test_prompt_has_no_banned_phrases(
    spec_a: PersonaSpec, spec_b: PersonaSpec, scenario_id: str, inject: bool
) -> None:
    scenario = SCENARIOS[scenario_id].for_pair(spec_a, spec_b)
    injected = spec_b.social.friction_points if inject else ()
    prompt = enactment_system_prompt(spec_a, scenario, "Sam", injected).lower()
    for phrase in (*BANNED_PHRASES, *EXTRA_BANNED):
        assert phrase not in prompt, phrase


def test_prompt_holds_friction_points_and_anti_sycophancy(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    prompt = enactment_system_prompt(spec_a, SCENARIOS["cold_open"], "Sam")
    assert "being talked over" in prompt
    assert "explaining my job" in prompt
    assert "Sam" in prompt
    assert "3 sentences" in prompt
    assert "disengage" in prompt
    assert "not helpful or pleasant by default" in prompt
    assert "friction points" in prompt
    assert "TONIGHT, SPECIFICALLY" not in prompt


def test_injected_friction_is_the_counterparts(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    prompt = enactment_system_prompt(
        spec_b, SCENARIOS["cold_open"], "Alex", spec_a.social.friction_points
    )
    assert "TONIGHT, SPECIFICALLY" in prompt
    assert "small talk about the weather" in prompt
    assert "Alex" in prompt


def test_prompt_is_deterministic(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    scenario = SCENARIOS["mild_friction"].for_pair(spec_a, spec_b)
    p1 = enactment_system_prompt(spec_a, scenario, "Sam")
    p2 = enactment_system_prompt(spec_a, scenario, "Sam")
    assert p1 == p2  # a stable prefix is what makes §8 caching work
