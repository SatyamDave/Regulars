from pathlib import Path

from regulars.core.persona import PersonaSpec
from regulars.sim.cache import InMemorySimCache, JsonlSimCache, SimCacheKey
from regulars.sim.harness import SimConfig, run_pair
from regulars.sim.scenarios import SCENARIOS
from tests.sim.conftest import echo_llm

CFG = SimConfig(turns=4, seeds=(0, 1))
SCENES = [SCENARIOS["cold_open"], SCENARIOS["dead_air"]]


async def test_in_memory_cache_hit_skips_calls(spec_a: PersonaSpec, spec_b: PersonaSpec) -> None:
    cache = InMemorySimCache()
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    first = await run_pair(llm_a, llm_b, spec_a, spec_b, CFG, scenarios=SCENES, cache=cache)
    calls_after_first = len(llm_a.calls) + len(llm_b.calls)
    assert calls_after_first == len(SCENES) * len(CFG.seeds) * CFG.turns
    assert len(cache) == len(SCENES) * len(CFG.seeds)

    second = await run_pair(llm_a, llm_b, spec_a, spec_b, CFG, scenarios=SCENES, cache=cache)
    assert len(llm_a.calls) + len(llm_b.calls) == calls_after_first  # no new calls
    assert second == first


async def test_cache_misses_on_persona_change_or_model_change(
    spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    cache = InMemorySimCache()
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    await run_pair(llm_a, llm_b, spec_a, spec_b, CFG, scenarios=SCENES, cache=cache)
    n = len(llm_a.calls)

    spec_b2 = spec_b.model_copy(
        update={"social": spec_b.social.model_copy(update={"drains": ["everything"]})}
    )
    await run_pair(llm_a, llm_b, spec_a, spec_b2, CFG, scenarios=SCENES, cache=cache)
    assert len(llm_a.calls) == 2 * n  # persona version bump recomputes

    other_model = echo_llm("C")
    await run_pair(llm_a, other_model, spec_a, spec_b, CFG, scenarios=SCENES, cache=cache)
    assert len(llm_a.calls) == 3 * n  # different model recomputes


async def test_jsonl_cache_round_trips_across_instances(
    tmp_path: Path, spec_a: PersonaSpec, spec_b: PersonaSpec
) -> None:
    path = tmp_path / "sims" / "cache.jsonl"
    cache = JsonlSimCache(path)
    llm_a, llm_b = echo_llm("A"), echo_llm("B")
    first = await run_pair(llm_a, llm_b, spec_a, spec_b, CFG, scenarios=SCENES, cache=cache)
    assert path.exists()
    assert len(path.read_text().splitlines()) == len(SCENES) * len(CFG.seeds)

    reloaded = JsonlSimCache(path)
    assert len(reloaded) == len(SCENES) * len(CFG.seeds)
    fresh_a, fresh_b = echo_llm("A"), echo_llm("B")
    second = await run_pair(fresh_a, fresh_b, spec_a, spec_b, CFG, scenarios=SCENES, cache=reloaded)
    assert not fresh_a.calls and not fresh_b.calls
    assert second == first


def test_jsonl_later_line_wins(tmp_path: Path) -> None:
    from regulars.core.transcript import Transcript, Turn

    path = tmp_path / "c.jsonl"
    key = SimCacheKey("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cold_open", 0, "m")
    t1 = Transcript(participants=["a", "b"], turns=[Turn(speaker_id="a", text="v1")], source="sim")
    t2 = Transcript(participants=["a", "b"], turns=[Turn(speaker_id="a", text="v2")], source="sim")
    cache = JsonlSimCache(path)
    cache.put(key, t1)
    cache.put(key, t2)
    assert JsonlSimCache(path).get(key) == t2
    assert JsonlSimCache(path).get(SimCacheKey("x", "y", "cold_open", 0, "m")) is None
