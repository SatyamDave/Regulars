from regulars.core.persona import PersonaSpec
from regulars.sim.scenarios import (
    DISAGREEMENT_PLACEHOLDER,
    FALLBACK_DISAGREEMENT,
    SCENARIOS,
    disagreement_for,
)


def test_three_scenarios_with_distinct_difficulty() -> None:
    assert set(SCENARIOS) == {"cold_open", "dead_air", "mild_friction"}
    assert len({s.difficulty for s in SCENARIOS.values()}) == 3
    assert [s.injects_friction for s in SCENARIOS.values()].count(True) == 1
    assert SCENARIOS["mild_friction"].injects_friction


def test_dead_air_carries_the_dying_exchange() -> None:
    dead_air = SCENARIOS["dead_air"]
    assert len(dead_air.seed_turns) >= 2
    assert not SCENARIOS["cold_open"].seed_turns


def test_mild_friction_names_disagreement_from_personas(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    template = SCENARIOS["mild_friction"]
    assert DISAGREEMENT_PLACEHOLDER in template.setup
    rendered = template.for_pair(spec_a, spec_b)
    assert DISAGREEMENT_PLACEHOLDER not in rendered.setup
    assert "being talked over" in rendered.setup
    assert rendered.id == template.id
    # persona B's friction point is used when A has none
    rendered_b = template.for_pair(PersonaSpec(), spec_b)
    assert "sarcasm at my expense" in rendered_b.setup


def test_disagreement_falls_back_to_escalates_on_then_default() -> None:
    only_escalates = PersonaSpec.model_validate({"content": {"escalates_on": ["tipping"]}})
    assert disagreement_for(only_escalates, PersonaSpec()) == "tipping"
    assert disagreement_for(PersonaSpec(), PersonaSpec()) == FALLBACK_DISAGREEMENT


def test_for_pair_is_identity_without_placeholder(spec_a: PersonaSpec) -> None:
    cold = SCENARIOS["cold_open"]
    assert cold.for_pair(spec_a, spec_a) is cold
