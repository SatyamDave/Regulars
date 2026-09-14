import sys
from pathlib import Path

import pytest

from regulars.persona.prosody import (
    AudioDependencyMissing,
    ProsodyFeatures,
    combine,
    count_laughter,
    extract_prosody,
    prosody_from_word_timestamps,
)


def _steady(n: int, wps: float, start: float = 0.0) -> list[tuple[str, float, float]]:
    """n words, evenly spaced, `wps` words per second, no gaps."""
    step = 1.0 / wps
    return [(f"w{i}", start + i * step, start + (i + 1) * step) for i in range(n)]


def test_speech_rate_and_duration() -> None:
    words = _steady(30, wps=3.0)  # 10 s of speech, 180 wpm
    p = prosody_from_word_timestamps(words)
    assert p.duration_s == pytest.approx(10.0)
    assert p.speech_rate_wpm == pytest.approx(180.0)
    assert p.pause_count == 0
    assert p.pause_mean_ms == 0.0
    assert p.pause_p90_ms == 0.0
    assert p.pitch_variance is None


def test_pauses_detected_above_threshold() -> None:
    a = _steady(5, wps=2.0)  # ends at 2.5
    b = _steady(5, wps=2.0, start=3.0)  # 500 ms gap
    c = _steady(5, wps=2.0, start=7.0)  # ends 5.5 -> starts 7.0 => 1500 ms gap
    d = _steady(2, wps=2.0, start=9.6)  # ends 9.5 -> 100 ms: articulation, not a pause
    p = prosody_from_word_timestamps(a + b + c + d)
    assert p.pause_count == 2
    assert p.pause_mean_ms == pytest.approx(1000.0)
    assert p.pause_p90_ms == pytest.approx(1500.0)


def test_unsorted_input_and_laughter() -> None:
    words = [("so", 1.0, 1.3), ("[laughter]", 1.4, 2.0), ("yeah", 0.0, 0.4), ("haha", 2.1, 2.5)]
    p = prosody_from_word_timestamps(words)
    assert p.laughter_count == 2
    assert p.duration_s == pytest.approx(2.5)
    # laughter tokens are not words
    assert p.speech_rate_wpm == pytest.approx(2 / 2.5 * 60)
    assert p.pause_count == 1  # 0.4 -> 1.0


def test_empty_and_single_word() -> None:
    assert prosody_from_word_timestamps([]) == ProsodyFeatures(
        speech_rate_wpm=0,
        pause_count=0,
        pause_mean_ms=0,
        pause_p90_ms=0,
        laughter_count=0,
        duration_s=0,
    )
    one = prosody_from_word_timestamps([("hi", 0.0, 0.0)])
    assert one.speech_rate_wpm == 0.0
    assert one.duration_s == 0.0


def test_count_laughter_lexicon() -> None:
    assert count_laughter(["Haha,", "(laughs)", "[LAUGHTER]", "lol", "hat", "ha"]) == 4


def test_combine_is_duration_weighted() -> None:
    fast = prosody_from_word_timestamps(_steady(30, wps=3.0))  # 10 s @180
    slow = prosody_from_word_timestamps(_steady(30, wps=1.0))  # 30 s @60
    both = combine([fast, slow])
    assert both is not None
    assert both.duration_s == pytest.approx(40.0)
    assert both.speech_rate_wpm == pytest.approx((180 * 10 + 60 * 30) / 40)
    assert combine([]) is None
    assert combine([prosody_from_word_timestamps([])]) is None


def test_extract_prosody_errors_clearly_without_librosa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "librosa", None)  # makes `import librosa` raise
    with pytest.raises(AudioDependencyMissing, match="uv sync --extra audio"):
        extract_prosody(tmp_path / "missing.wav")


def test_extract_prosody_on_synthetic_wav(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 180 * t)
    silence = np.zeros(sr)  # 1 s pause
    y = np.concatenate([tone, silence, tone]).astype("float32")
    wav = tmp_path / "tone.wav"
    sf.write(wav, y, sr)

    p = extract_prosody(wav)
    assert p.duration_s == pytest.approx(3.0, abs=0.01)
    assert p.pause_count >= 1
    assert p.pause_p90_ms >= 800
    assert p.pitch_variance is not None

    with_words = extract_prosody(wav, words=_steady(6, wps=2.0))
    assert with_words.speech_rate_wpm == pytest.approx(120.0)
    assert with_words.duration_s == pytest.approx(3.0, abs=0.01)
