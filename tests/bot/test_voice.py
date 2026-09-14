"""The voice is the product (product-plan, Open items). These tests hold the style guide."""

from __future__ import annotations

import unicodedata

from regulars.bot import voice


def all_strings() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in vars(voice).items():
        if name.startswith("_") or name == "BANNED_PHRASES":
            continue
        if isinstance(value, str):
            out[name] = value
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    out[f"{name}[{i}]"] = item
    return out


def is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So" or ord(ch) >= 0x1F000


def test_voice_module_has_strings() -> None:
    assert len(all_strings()) > 20


def test_no_exclamation_marks_or_emojis() -> None:
    for name, text in all_strings().items():
        assert "!" not in text, name
        assert not any(is_emoji(ch) for ch in text), name


def test_banned_phrases_absent() -> None:
    for name, text in all_strings().items():
        lowered = text.lower()
        for phrase in voice.BANNED_PHRASES:
            assert phrase not in lowered, (name, phrase)


def test_style_guide_is_documented() -> None:
    assert voice.__doc__ is not None
    for word in ("Short", "warm", "Specific", "hype", "exclamation", "emojis"):
        assert word.lower() in voice.__doc__.lower()
