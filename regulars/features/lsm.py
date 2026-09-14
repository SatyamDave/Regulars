"""Linguistic style matching (docs/technical-architecture.md §4).

Ireland & Pennebaker's nine function-word categories, each as a frozen lexicon. Per category::

    LSM_c = 1 - |p_a,c - p_b,c| / (p_a,c + p_b,c + 0.0001)
    LSM   = mean over the nine categories

``p`` is the share of a speaker's tokens in the category, expressed as a percentage (0-100) as
LIWC reports it, so the 0.0001 smoothing constant keeps the scale the formula was written for.

Slope matters more than level: ``lsm_trajectory`` / ``lsm_slope`` measure convergence over the
conversation; two people who happen to start similar is not the signal.

CAVEAT (hold it honestly): Ireland et al. (2011) found LSM predicted romantic interest and
relationship stability in speed dating, but Niederhoffer & Pennebaker found LSM unrelated to
participant- and judge-rated interaction quality and argued it indexes *engagement*, not
rapport. LSM is a FEATURE for the judge, NEVER a TARGET to optimize or a proxy for a good
conversation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from regulars.core.transcript import Transcript, Turn
from regulars.features._text import tokenize, words

AUXILIARY_VERBS: frozenset[str] = words(
    """
    am is are was were be been being have has had having do does did doing will would shall
    should can could may might must ought i'm i've i'd i'll you're you've you'd you'll he's
    she's it's we're we've we'd we'll they're they've they'd they'll that's there's isn't aren't
    wasn't weren't haven't hasn't hadn't don't doesn't didn't won't wouldn't can't couldn't
    shouldn't mustn't ain't gonna gotta wanna
    """
)
ARTICLES: frozenset[str] = words("a an the")
COMMON_ADVERBS: frozenset[str] = words(
    """
    very really just so too quite rather pretty almost also even still already always often
    sometimes usually again here there now then soon maybe probably definitely actually basically
    especially exactly totally absolutely well ever far only once twice kinda sorta literally
    honestly seriously anyway somewhat truly barely hardly
    """
)
PERSONAL_PRONOUNS: frozenset[str] = words(
    """
    i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his
    himself she her hers herself they them their theirs themselves ya y'all i'm i've i'd i'll
    you're you've you'd you'll we're we've we'd we'll he's she's they're they've they'd they'll
    """
)
IMPERSONAL_PRONOUNS: frozenset[str] = words(
    """
    it its itself it's this that these those that's there's something anything nothing
    everything someone anyone everyone everybody somebody anybody nobody what which who whom
    whose whatever whoever whichever other others another none
    """
)
PREPOSITIONS: frozenset[str] = words(
    """
    about above across after against along among around at before behind below beneath beside
    between beyond by despite down during except for from in inside into like near of off on
    onto out outside over past since through throughout till to toward towards under underneath
    until up upon with within without
    """
)
NEGATIONS: frozenset[str] = words(
    """
    no not never none nothing nobody nowhere neither nor cannot can't don't doesn't didn't won't
    wouldn't shouldn't couldn't isn't aren't wasn't weren't haven't hasn't hadn't ain't mustn't
    needn't nope nah
    """
)
CONJUNCTIONS: frozenset[str] = words(
    """
    and but or nor so yet because although though while whereas unless until since if whether as
    than when whenever where wherever after before once plus either neither both
    """
)
QUANTIFIERS: frozenset[str] = words(
    """
    all any both each every few fewer fewest less least little lot lots many more most much none
    several some enough plenty half whole entire numerous various majority minority couple dozens
    hundreds thousands tons bunch bit single double
    """
)

CATEGORIES: dict[str, frozenset[str]] = {
    "auxiliary_verbs": AUXILIARY_VERBS,
    "articles": ARTICLES,
    "common_adverbs": COMMON_ADVERBS,
    "personal_pronouns": PERSONAL_PRONOUNS,
    "impersonal_pronouns": IMPERSONAL_PRONOUNS,
    "prepositions": PREPOSITIONS,
    "negations": NEGATIONS,
    "conjunctions": CONJUNCTIONS,
    "quantifiers": QUANTIFIERS,
}

SMOOTHING = 0.0001


def category_rates(text: str) -> dict[str, float]:
    """Percentage of tokens in each category (a word may fall in several, as in LIWC)."""
    tokens = tokenize(text)
    if not tokens:
        return dict.fromkeys(CATEGORIES, 0.0)
    n = len(tokens)
    return {
        cat: 100.0 * sum(tok in lexicon for tok in tokens) / n
        for cat, lexicon in CATEGORIES.items()
    }


def lsm_by_category(text_a: str, text_b: str) -> dict[str, float]:
    ra, rb = category_rates(text_a), category_rates(text_b)
    return {c: 1.0 - abs(ra[c] - rb[c]) / (ra[c] + rb[c] + SMOOTHING) for c in CATEGORIES}


def lsm(text_a: str, text_b: str) -> float:
    """Mean LSM over the nine categories; 0.0 if either side has no tokens."""
    if not tokenize(text_a) or not tokenize(text_b):
        return 0.0
    per = lsm_by_category(text_a, text_b)
    return sum(per.values()) / len(per)


def _speaker_texts(turns: Sequence[Turn], a: str, b: str) -> tuple[str, str]:
    ta = " ".join(t.text for t in turns if t.speaker_id == a)
    tb = " ".join(t.text for t in turns if t.speaker_id == b)
    return ta, tb


def lsm_mean(transcript: Transcript) -> float:
    """Aggregate LSM between the two participants' pooled text."""
    a, b = transcript.participants[0], transcript.participants[1]
    return lsm(*_speaker_texts(transcript.spoken(), a, b))


def lsm_trajectory(transcript: Transcript, window_turns: int = 6) -> list[float]:
    """LSM in a sliding window (step 1) over spoken turns.

    Windows in which only one participant speaks are skipped. A transcript shorter than the
    window yields a single value over all of it (or nothing if one side never spoke).
    """
    if window_turns < 2:
        raise ValueError("window_turns must be >= 2")
    turns = transcript.spoken()
    a, b = transcript.participants[0], transcript.participants[1]
    if len(turns) <= window_turns:
        windows = [turns] if turns else []
    else:
        windows = [turns[i : i + window_turns] for i in range(len(turns) - window_turns + 1)]
    out: list[float] = []
    for w in windows:
        speakers = {t.speaker_id for t in w}
        if a in speakers and b in speakers:
            out.append(lsm(*_speaker_texts(w, a, b)))
    return out


def lsm_slope(trajectory: Sequence[float]) -> float:
    """Least-squares slope of LSM against window index; 0.0 with fewer than two points."""
    if len(trajectory) < 2:
        return 0.0
    x = np.arange(len(trajectory), dtype=float)
    y = np.asarray(trajectory, dtype=float)
    x_c = x - x.mean()
    denom = float(np.dot(x_c, x_c))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_c, y - y.mean()) / denom)


__all__ = [
    "ARTICLES",
    "AUXILIARY_VERBS",
    "CATEGORIES",
    "COMMON_ADVERBS",
    "CONJUNCTIONS",
    "IMPERSONAL_PRONOUNS",
    "NEGATIONS",
    "PERSONAL_PRONOUNS",
    "PREPOSITIONS",
    "QUANTIFIERS",
    "SMOOTHING",
    "category_rates",
    "lsm",
    "lsm_by_category",
    "lsm_mean",
    "lsm_slope",
    "lsm_trajectory",
]
