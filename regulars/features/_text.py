"""Shared lexical helpers for the heuristic classifiers (private to regulars.features).

Deliberately tiny: a regex tokenizer, a stopword list, a suffix stemmer and Jaccard. The
heuristics are the cheap first pass (docs/technical-architecture.md §4); the LLM-backed
implementations do the judgment calls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def words(block: str) -> frozenset[str]:
    """Whitespace-separated word list -> frozen lexicon (keeps the lexicons readable)."""
    return frozenset(block.split())


STOPWORDS: frozenset[str] = words(
    """
    a about above across after again against ain't all almost also although always am an and
    another any anybody anyone anything are aren't around as at away back be because been
    before behind being below beside between both but by can can't cannot could couldn't did
    didn't do does doesn't doing don't down during each either enough even ever every everybody
    everyone everything few for from get gets getting go goes going gonna got gotta had hadn't
    has hasn't have haven't having he he's hello her here hers herself hey hi him himself his
    how how's i i'd i'll i'm i've if in into is isn't it it's its itself just kind know let let's
    like lot lots many may maybe me mean might mine more most much must my myself never no nobody
    none nope nor not nothing now of off oh ok okay on once one only or other others ought our
    ours ourselves out over own pretty quite rather really right said same say see shall she
    she's should shouldn't since so some somebody someone something sort still such sure take
    than that that's the their theirs them themselves then there there's these they they'd
    they'll they're they've thing things this those though through till to too totally under
    until up upon us very wanna want was wasn't way we we'd we'll we're we've well were weren't
    what what's whatever when where where's whether which while who who's whom whose why why's
    will with within without won't would wouldn't ya yeah yep yes yet you you'd you'll you're
    you've your yours yourself yourselves
    """
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; internal apostrophes kept so contractions stay one token."""
    return _TOKEN_RE.findall(text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def stem(word: str) -> str:
    """Crude suffix stripper: enough to make ``climbing`` and ``climbed`` share a lemma."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix) and not w.endswith("ss"):
            return w[: -len(suffix)]
    return w


def content_lemmas(text: str) -> set[str]:
    """Stemmed non-stopword tokens of length >= 3."""
    out: set[str] = set()
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        s = stem(tok)
        if len(s) >= 3:
            out.add(s)
    return out


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
