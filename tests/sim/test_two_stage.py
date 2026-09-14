from regulars.core.persona import PersonaSpec
from regulars.core.transcript import Transcript
from regulars.sim.cache import InMemorySimCache
from regulars.sim.harness import SimConfig
from regulars.sim.scenarios import SCENARIOS
from regulars.sim.two_stage import rank_candidates
from tests.sim.conftest import echo_llm

CFG = SimConfig(turns=4, seeds=(0, 1, 2, 3, 4))


def candidate(i: int) -> PersonaSpec:
    return PersonaSpec.model_validate({"social": {"friction_points": [f"thing {i}"]}})


def score_by_candidate(transcripts: list[Transcript]) -> float:
    """Deterministic stand-in for the judge: candidate 'c<i>' scores i, plus a bonus when the
    full grid ran, so the frontier pass is distinguishable from the cheap pass."""
    cid = transcripts[0].participants[1]
    base = float(cid[1:])
    return base + (0.5 if len(transcripts) > 1 else 0.0)


async def test_two_stage_call_counts_and_ordering(spec_a: PersonaSpec) -> None:
    cheap, frontier = echo_llm("cheap"), echo_llm("frontier")
    candidates = {f"c{i}": candidate(i) for i in range(5)}
    ranked = await rank_candidates(
        spec_a, candidates, cheap, frontier, score_by_candidate, first_pass_top_k=2, config=CFG
    )
    n_scen = len(SCENARIOS)
    # cheap pass: every candidate, 1 scenario x 1 seed, both sides on the cheap model
    assert len(cheap.calls) == 5 * CFG.turns
    # frontier pass: top-k only, full 3 scenarios x 5 seeds
    assert len(frontier.calls) == 2 * n_scen * len(CFG.seeds) * CFG.turns

    assert [r.candidate_id for r in ranked] == ["c4", "c3", "c2", "c1", "c0"]
    assert [r.frontier_score is not None for r in ranked] == [True, True, False, False, False]
    assert ranked[0].frontier_score == 4.5 and ranked[0].cheap_score == 4.0
    assert ranked[0].score == 4.5 and ranked[2].score == 2.0
    assert len(ranked[0].transcripts) == n_scen * len(CFG.seeds)
    assert len(ranked[2].transcripts) == 1
    # cheap-pass transcripts came from the cheap model and ran on the friction seed
    cheap_t = ranked[2].transcripts[0]
    assert cheap_t.metadata["model_a"] == "cheap"
    assert cheap_t.metadata["seed"] == CFG.friction_seed
    assert cheap_t.metadata["friction_injected"] == 1
    assert cheap_t.participants == ["user", "c2"]


async def test_top_k_larger_than_pool_runs_frontier_on_all(spec_a: PersonaSpec) -> None:
    cheap, frontier = echo_llm("cheap"), echo_llm("frontier")
    candidates = {f"c{i}": candidate(i) for i in range(3)}
    ranked = await rank_candidates(
        spec_a, candidates, cheap, frontier, score_by_candidate, first_pass_top_k=40, config=CFG
    )
    assert all(r.frontier_score is not None for r in ranked)
    assert len(frontier.calls) == 3 * len(SCENARIOS) * len(CFG.seeds) * CFG.turns


async def test_top_k_zero_skips_frontier(spec_a: PersonaSpec) -> None:
    cheap, frontier = echo_llm("cheap"), echo_llm("frontier")
    ranked = await rank_candidates(
        spec_a, {"c1": candidate(1)}, cheap, frontier, score_by_candidate, 0, config=CFG
    )
    assert not frontier.calls
    assert ranked[0].frontier_score is None


async def test_empty_candidates(spec_a: PersonaSpec) -> None:
    assert await rank_candidates(spec_a, {}, echo_llm(), echo_llm(), score_by_candidate) == []


async def test_cache_is_consulted_in_both_stages(spec_a: PersonaSpec) -> None:
    cache = InMemorySimCache()
    candidates = {f"c{i}": candidate(i) for i in range(3)}
    cheap, frontier = echo_llm("cheap"), echo_llm("frontier")
    await rank_candidates(
        spec_a, candidates, cheap, frontier, score_by_candidate, 1, config=CFG, cache=cache
    )
    cheap2, frontier2 = echo_llm("cheap"), echo_llm("frontier")
    ranked = await rank_candidates(
        spec_a, candidates, cheap2, frontier2, score_by_candidate, 1, config=CFG, cache=cache
    )
    assert not cheap2.calls and not frontier2.calls
    assert ranked[0].candidate_id == "c2"
