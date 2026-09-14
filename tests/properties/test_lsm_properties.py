"""Property-based tests for regulars/features/lsm.py.

For any two generated texts:

* ``lsm(a, b) == lsm(b, a)`` (symmetry -- the formula uses absolute difference).
* ``lsm(a, a)`` (identical text) is always its maximum possible value, 1.0 (each category's
  ``|p_a,c - p_b,c|`` term is 0 when a and b are the same text, so every category scores 1.0).
* the result is always in the bounded range implied by the formula::

      LSM_c = 1 - |p_a,c - p_b,c| / (p_a,c + p_b,c + 0.0001)

  Since p is a percentage in [0, 100] and the numerator is <= the (non-smoothed) denominator, each
  category term is in (0, 1], with a slight downward slack from the smoothing constant, so the
  mean over categories (``lsm``) is always in [0, 1].
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from regulars.features.lsm import CATEGORIES, lsm, lsm_by_category

# Build texts from a mix of category words (so categories actually fire) and filler words.
_category_words = sorted({w for lexicon in CATEGORIES.values() for w in lexicon})
_filler_words = ["banana", "zephyr", "quokka", "gizmo", "widget"]
_word_pool = _category_words + _filler_words

_text = st.lists(st.sampled_from(_word_pool), min_size=0, max_size=20).map(" ".join)


@given(a=_text, b=_text)
@settings(max_examples=200)
def test_lsm_symmetric(a: str, b: str) -> None:
    assert lsm(a, b) == lsm(b, a)


@given(a=_text, b=_text)
@settings(max_examples=200)
def test_lsm_by_category_symmetric(a: str, b: str) -> None:
    per_ab = lsm_by_category(a, b)
    per_ba = lsm_by_category(b, a)
    assert per_ab == per_ba


@given(text=_text)
@settings(max_examples=200)
def test_lsm_identical_text_is_maximum(text: str) -> None:
    value = lsm(text, text)
    if text.strip() == "" or not any(c.isalnum() for c in text):
        # tokenize() yields no tokens -> lsm() returns 0.0 by the documented empty-text rule.
        assert value == 0.0
    else:
        assert value == 1.0


@given(a=_text, b=_text)
@settings(max_examples=200)
def test_lsm_bounded_in_unit_interval(a: str, b: str) -> None:
    value = lsm(a, b)
    assert 0.0 <= value <= 1.0
    for cat_value in lsm_by_category(a, b).values():
        assert 0.0 <= cat_value <= 1.0
