from collections.abc import Sequence

import pytest

from regulars.core.llm import FakeLLM, LLMMessage
from regulars.core.persona import PersonaSpec
from regulars.sim.harness import OPENING_CUE, SimConfig, persona_hash, run_pair, run_sim
from regulars.sim.scenarios import SCENARIOS
from tests.sim.conftest import echo_llm

SMALL = SimConfig(turns=6, seeds=(0, 1, 2), temperature=0.95, friction_seed_index=1)


def alternates(speakers: list[str]) -> bool:
    return all(x != y for x, y in zip(speakers, speakers[1:], strict=False))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
async def test_exact_turn_count_and_alternation(
    spec_a: PersonaSpec, spec_b: PersonaSpec, seed: int
) -> None:
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    t = await run_sim(llm_a, llm_b, spec_a, spec_b, SCENARIOS["cold_open"], seed, SMALL)
    speakers = [turn.speaker_id for turn in t.turns]
    assert len(t.turns) == SMALL.turns
    assert alternates(speakers)
    assert speakers[0] == ("a" if seed % 2 == 0 else "b")  # A opens on even seeds
    assert len(llm_a.calls) + len(llm_b.calls) == SMALL.turns
    assert len(llm_a.calls) == len(llm_b.calls) == SMALL.turns // 2
    assert t.source == "sim"
    assert all(turn.sent_at_ms is None for turn in t.turns)


async def test_dead_air_seed_turns_prefix_the_transcript(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    dead_air = SCENARIOS["dead_air"]
    t = await run_sim(echo_llm(), echo_llm(), spec_a, spec_b, dead_air, 0, SMALL)
    n_seed = len(dead_air.seed_turns)
    assert len(t.turns) == n_seed + SMALL.turns
    assert [x.text for x in t.turns[:n_seed]] == list(dead_air.seed_turns)
    assert alternates([x.speaker_id for x in t.turns])
    assert t.metadata["seed_turns"] == n_seed


async def test_history_is_from_each_personas_pov(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    t = await run_sim(llm_a, llm_b, spec_a, spec_b, SCENARIOS["cold_open"], 0, SMALL)
    # A's opening call: no history, so it gets the neutral cue as the required user message
    first_system, first_msgs = llm_a.calls[0]
    assert [m.role for m in first_msgs] == ["user"]
    assert first_msgs[0].content == OPENING_CUE
    assert "being talked over" in first_system  # A sees its own spec
    # B's second call sees A, B, A as user/assistant/user
    _, b_msgs = llm_b.calls[1]
    assert [m.role for m in b_msgs] == ["user", "assistant", "user"]
    assert [m.content for m in b_msgs] == [t.turns[0].text, t.turns[1].text, t.turns[2].text]
    # every call ends on a user message and the roles alternate strictly
    for llm in (llm_a, llm_b):
        for _, msgs in llm.calls:
            assert msgs[-1].role == "user"
            assert alternates([m.role for m in msgs])


async def test_metadata(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    t = await run_sim(
        echo_llm("modelA"), echo_llm("modelB"), spec_a, spec_b, SCENARIOS["cold_open"], 3, SMALL
    )
    md = t.metadata
    assert md["scenario_id"] == "cold_open"
    assert md["seed"] == 3
    assert md["persona_hash_a"] == persona_hash(spec_a)
    assert md["persona_hash_b"] == persona_hash(spec_b)
    assert len(str(md["persona_hash_a"])) == 12
    assert md["persona_hash_a"] != md["persona_hash_b"]
    assert md["model_a"] == "modelA"
    assert md["model_b"] == "modelB"
    assert md["temperature"] == 0.95
    assert md["friction_injected"] == 0
    assert md["opener"] == "b"


def test_persona_hash_is_stable_and_sensitive(spec_a: PersonaSpec) -> None:
    assert persona_hash(spec_a) == persona_hash(spec_a.model_copy(deep=True))
    changed_stated = spec_a.stated.model_copy(update={"seeking": "x"})
    changed = spec_a.model_copy(update={"stated": changed_stated})
    assert persona_hash(changed) != persona_hash(spec_a)


async def test_friction_injected_for_exactly_one_seed_per_pair(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    transcripts = await run_pair(llm_a, llm_b, spec_a, spec_b, SMALL)
    assert len(transcripts) == len(SCENARIOS) * len(SMALL.seeds)
    injected_seeds = {t.metadata["seed"] for t in transcripts if t.metadata["friction_injected"]}
    assert injected_seeds == {SMALL.seeds[SMALL.friction_seed_index]}
    # each scenario has exactly one injected seed
    for scenario_id in SCENARIOS:
        flags = [
            t.metadata["friction_injected"]
            for t in transcripts
            if t.metadata["scenario_id"] == scenario_id
        ]
        assert sum(flags) == 1
    # on the friction seed, B's prompt carries A's friction points and A's carries B's
    b_systems_with_a_friction = [s for s, _ in llm_b.calls if "small talk about the weather" in s]
    a_systems_with_b_friction = [s for s, _ in llm_a.calls if "sarcasm at my expense" in s]
    # 3 scenarios x (turns/2) calls each on the injected seed only
    assert len(b_systems_with_a_friction) == len(SCENARIOS) * (SMALL.turns // 2)
    assert len(a_systems_with_b_friction) == len(SCENARIOS) * (SMALL.turns // 2)
    # results ordered scenario-major, seed-minor
    assert [(t.metadata["scenario_id"], t.metadata["seed"]) for t in transcripts] == [
        (sid, seed) for sid in SCENARIOS for seed in SMALL.seeds
    ]


async def test_run_pair_passes_temperature_and_seed(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    seen: list[tuple[float, int | None, int]] = []

    class Spy:
        model = "spy"

        async def complete(
            self,
            system: str,
            messages: Sequence[LLMMessage],
            *,
            temperature: float = 1.0,
            max_tokens: int = 512,
            seed: int | None = None,
        ) -> str:
            seen.append((temperature, seed, max_tokens))
            return "ok"

    cfg = SimConfig(turns=2, seeds=(7,), temperature=0.9, max_tokens=99)
    await run_pair(Spy(), Spy(), spec_a, spec_b, cfg, scenarios=[SCENARIOS["cold_open"]])
    assert seen == [(0.9, 7, 99), (0.9, 7, 99)]


async def test_name_prefix_is_stripped(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    # turn order on seed 0 is Alex, Sam, Alex; only the speaker's own prefix is stripped
    llm = FakeLLM(replies=["Alex: hi there", "  ", "Alex: yo"])
    cfg = SimConfig(turns=3, seeds=(0,))
    t = await run_sim(
        llm, llm, spec_a, spec_b, SCENARIOS["cold_open"], 0, cfg, participants=("Alex", "Sam")
    )
    assert [x.text for x in t.turns] == ["hi there", "...", "yo"]
    assert t.participants == ["Alex", "Sam"]
