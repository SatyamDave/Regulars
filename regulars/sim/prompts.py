"""Enactment system prompt (§3 anti-sycophancy).

The prompt is the per-persona prefix cached across every sim for that persona (§8), so it must
be a pure function of (spec, scenario, counterpart name, injected friction). It must never carry
any instruction to have a good conversation, be pleasant, or build a connection: two LLMs told
that are pleasant to each other and the sim has no variance (§6, problem 4). ``BANNED_PHRASES``
is the list the unit test enforces.
"""

from __future__ import annotations

from collections.abc import Sequence

from regulars.core.persona import PersonaSpec
from regulars.sim.scenarios import Scenario

# Lower-cased substrings that must never appear in an enactment prompt.
BANNED_PHRASES: tuple[str, ...] = (
    "good conversation",
    "great conversation",
    "rapport",
    "be helpful",
    "be pleasant",
    "be friendly",
    "be warm",
    "be kind",
    "be polite",
    "be engaging",
    "keep the conversation going",
    "keep things going",
    "find common ground",
    "common ground",
    "make them feel",
    "get to know",
    "connect with",
    "connection",
    "enjoyable",
    "positive",
    "hit it off",
)

MAX_SENTENCES_PER_TURN = 3


def _bullets(items: Sequence[str], empty: str = "none recorded") -> str:
    clean = [i.strip() for i in items if i.strip()]
    if not clean:
        return f"  - ({empty})"
    return "\n".join(f"  - {i}" for i in clean)


def render_spec(spec: PersonaSpec) -> str:
    """The PersonaSpec as plain text. Provenance is omitted: it is bookkeeping, not behavior."""
    c, k, s, st = spec.conversational, spec.content, spec.social, spec.stated
    lines = [
        "HOW YOU TALK",
        f"  opener style: {c.opener_style or 'unremarkable'}",
        f"  typical turn length: {c.avg_turn_length}",
        f"  humor: {c.humor_register}",
        f"  asks questions: {c.question_tendency}",
        f"  interrupting: {c.interrupt_tendency or 'unremarkable'}",
        f"  silence tolerance: {c.silence_tolerance or 'unremarkable'}",
        "WHAT YOU TALK ABOUT",
        "  brings up unprompted:",
        _bullets(k.unprompted_topics),
        "  avoids:",
        _bullets(k.avoided_topics),
        "  gets heated about:",
        _bullets(k.escalates_on),
        "  deflects on:",
        _bullets(k.deflects_on),
        "HOW YOU ARE WITH PEOPLE",
        f"  energy: {s.energy}",
        f"  opens up: {s.disclosure_speed}",
        "  things that rub you the wrong way (friction points):",
        _bullets(s.friction_points),
        "  things that drain you:",
        _bullets(s.drains),
        "WHAT YOU SAID YOU WANT",
        f"  seeking: {st.seeking or 'did not say'}",
        "  will not do:",
        _bullets(st.exclusions),
    ]
    return "\n".join(lines)


def enactment_system_prompt(
    spec: PersonaSpec,
    scenario: Scenario,
    counterpart_first_name: str,
    friction_injected: Sequence[str] = (),
) -> str:
    """System prompt for one persona in one scene.

    ``friction_injected`` is the *counterpart's* friction_points (§3: injected into this persona's
    scene for one seed per pair). They are surfaced as things this persona happens to do tonight,
    so the friction actually lands instead of staying hypothetical.
    """
    scene = scenario.setup.replace("{counterpart}", counterpart_first_name)
    parts = [
        "You are a real person at a dinner arranged for strangers. You are not an assistant and "
        "nobody here is your customer. Stay in character for the whole scene.",
        "",
        "WHO YOU ARE",
        render_spec(spec),
        "",
        "THE SCENE",
        scene,
        "",
        "HOW TO PLAY IT",
        "  - Speak only as yourself. Output your spoken words only: no name prefix, no stage "
        "directions, no narration, no quotation marks around the whole line.",
        f"  - At most {MAX_SENTENCES_PER_TURN} sentences per turn. A single word is a fine turn.",
        "  - You are not helpful or pleasant by default. You are exactly as interested as this "
        "person would be, and no more.",
        "  - When you are bored, disengage: answer short, change nothing, let the silence sit, "
        "check out. Do not manufacture interest you do not have.",
        "  - When a topic is one of your drains, give it as little as the real you would.",
        "  - Hold your friction points. If the other person does one of the things that rubs "
        "you wrong, react to it the way you actually would, in the moment.",
        "  - Do not ask a question every turn. Ask only when the real you would want the answer.",
        "  - Do not summarize, reflect back, or validate what they said unless that is "
        "genuinely how you talk.",
        "  - Disagreement does not need resolving. You may leave it there.",
    ]
    injected = [f.strip() for f in friction_injected if f.strip()]
    if injected:
        parts += [
            "",
            "TONIGHT, SPECIFICALLY",
            f"  {counterpart_first_name} has described these, in their own words, as things "
            "they cannot stand. Without meaning to and without announcing it, over the course "
            "of the scene you do them anyway:",
            _bullets(injected),
        ]
    return "\n".join(parts)
