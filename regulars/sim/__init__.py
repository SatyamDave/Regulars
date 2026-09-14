"""Persona × persona simulation harness, scenarios, anti-sycophancy prompt, cache, two-stage
ranking and the variance diagnostic (§3, §6 problem 4, §8)."""

from regulars.sim.cache import InMemorySimCache, JsonlSimCache, SimCache, SimCacheKey
from regulars.sim.harness import SimConfig, persona_hash, run_pair, run_sim
from regulars.sim.prompts import BANNED_PHRASES, enactment_system_prompt, render_spec
from regulars.sim.scenarios import (
    COLD_OPEN,
    DEAD_AIR,
    MILD_FRICTION,
    SCENARIOS,
    Scenario,
    disagreement_for,
)
from regulars.sim.two_stage import RankedCandidate, rank_candidates
from regulars.sim.variance import VarianceReport, sim_variance_report

__all__ = [
    "BANNED_PHRASES",
    "COLD_OPEN",
    "DEAD_AIR",
    "MILD_FRICTION",
    "SCENARIOS",
    "InMemorySimCache",
    "JsonlSimCache",
    "RankedCandidate",
    "Scenario",
    "SimCache",
    "SimCacheKey",
    "SimConfig",
    "VarianceReport",
    "disagreement_for",
    "enactment_system_prompt",
    "persona_hash",
    "rank_candidates",
    "render_spec",
    "run_pair",
    "run_sim",
    "sim_variance_report",
]
