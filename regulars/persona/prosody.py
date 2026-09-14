"""Prosody features from voice-memo audio (§2 Extraction).

Speech rate and pause structure are the only pre-meeting proxy for the response-latency signal
that dominates the judge (§5), so they are kept alongside the transcript rather than discarded.

Two entry points:

* :func:`prosody_from_word_timestamps` — pure Python over Whisper word timestamps. No audio
  dependencies; this is the path the bot uses and the one that is tested.
* :func:`extract_prosody` — reads a WAV with ``librosa`` (the ``audio`` extra). The import is
  lazy so this module always imports; the function raises :class:`AudioDependencyMissing` when
  the extra is not installed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

WordTimestamp = tuple[str, float, float]
"""``(word, start_s, end_s)`` as emitted by Whisper with ``timestamp_granularities=["word"]``."""

PAUSE_THRESHOLD_MS = 300.0
"""Inter-word gaps shorter than this are articulation, not pauses."""

_LAUGHTER = re.compile(
    r"^\W*(\[laugh(s|ter|ing)?\]|\(laugh(s|ter|ing)?\)|ha(ha)+|lol|lmao)\W*$", re.I
)


class ProsodyFeatures(BaseModel):
    """Audio-derived pacing features for one speaker across one or more recordings."""

    speech_rate_wpm: float = Field(ge=0)
    pause_count: int = Field(ge=0)
    pause_mean_ms: float = Field(ge=0)
    pause_p90_ms: float = Field(ge=0)
    pitch_variance: float | None = Field(default=None, ge=0)
    """Variance of voiced f0 in Hz². ``None`` when only text timestamps were available."""
    laughter_count: int = Field(ge=0)
    duration_s: float = Field(ge=0)


class AudioDependencyMissing(RuntimeError):
    """Raised when audio-based extraction is requested without the ``audio`` extra installed."""


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[rank]


def _pauses_ms(intervals: Sequence[tuple[float, float]]) -> list[float]:
    gaps: list[float] = []
    for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:], strict=False):
        gap_ms = (next_start - prev_end) * 1000.0
        if gap_ms >= PAUSE_THRESHOLD_MS:
            gaps.append(gap_ms)
    return gaps


def count_laughter(words: Sequence[str]) -> int:
    """Count laughter tokens Whisper emits (``[laughter]``, ``haha``, ...)."""
    return sum(1 for w in words if _LAUGHTER.match(w))


def prosody_from_word_timestamps(words: Sequence[WordTimestamp]) -> ProsodyFeatures:
    """Speech rate, pauses and laughter from word timestamps alone.

    ``duration_s`` is the span from the first word's start to the last word's end. Laughter
    tokens are excluded from the word count. Pitch cannot be derived from text, so
    ``pitch_variance`` is ``None``.
    """
    if not words:
        return ProsodyFeatures(
            speech_rate_wpm=0.0,
            pause_count=0,
            pause_mean_ms=0.0,
            pause_p90_ms=0.0,
            laughter_count=0,
            duration_s=0.0,
        )
    ordered = sorted(words, key=lambda w: w[1])
    duration_s = max(0.0, ordered[-1][2] - ordered[0][1])
    gaps = _pauses_ms([(s, e) for _, s, e in ordered])
    spoken = [w for w, _, _ in ordered]
    laughter = count_laughter(spoken)
    lexical = len(spoken) - laughter
    rate = (lexical / duration_s) * 60.0 if duration_s > 0 else 0.0
    return ProsodyFeatures(
        speech_rate_wpm=rate,
        pause_count=len(gaps),
        pause_mean_ms=sum(gaps) / len(gaps) if gaps else 0.0,
        pause_p90_ms=_percentile(gaps, 0.9),
        laughter_count=laughter,
        duration_s=duration_s,
    )


def combine(features: Sequence[ProsodyFeatures]) -> ProsodyFeatures | None:
    """Duration-weighted merge of per-recording features into one speaker profile."""
    usable = [f for f in features if f.duration_s > 0]
    if not usable:
        return None
    total = sum(f.duration_s for f in usable)
    pauses = sum(f.pause_count for f in usable)
    pitched = [f for f in usable if f.pitch_variance is not None]
    return ProsodyFeatures(
        speech_rate_wpm=sum(f.speech_rate_wpm * f.duration_s for f in usable) / total,
        pause_count=pauses,
        pause_mean_ms=(
            sum(f.pause_mean_ms * f.pause_count for f in usable) / pauses if pauses else 0.0
        ),
        pause_p90_ms=max(f.pause_p90_ms for f in usable),
        pitch_variance=(
            sum((f.pitch_variance or 0.0) * f.duration_s for f in pitched)
            / sum(f.duration_s for f in pitched)
            if pitched
            else None
        ),
        laughter_count=sum(f.laughter_count for f in usable),
        duration_s=total,
    )


def extract_prosody(
    wav_path: str | Path, words: Sequence[WordTimestamp] | None = None
) -> ProsodyFeatures:
    """Prosody from audio via ``librosa``.

    When ``words`` (Whisper timestamps) are supplied, speech rate, pauses and laughter come from
    them and only ``pitch_variance`` and ``duration_s`` come from the audio. Without them, pauses
    are the silences between non-silent regions and speech rate is estimated from onset density
    (roughly one onset per syllable, ~1.5 syllables per English word); treat that estimate as
    coarse.
    """
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise AudioDependencyMissing(
            "extract_prosody needs librosa/soundfile; install with `uv sync --extra audio`"
        ) from exc

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration_s = float(len(y) / sr) if sr else 0.0

    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    n = min(len(f0), len(rms))
    voiced = f0[:n][rms[:n] > 0.1 * float(rms.max() or 1.0)]
    pitch_variance = float(np.var(voiced)) if len(voiced) > 1 else 0.0

    if words:
        from_words = prosody_from_word_timestamps(words)
        return from_words.model_copy(
            update={"pitch_variance": pitch_variance, "duration_s": duration_s}
        )

    intervals = [(float(s) / sr, float(e) / sr) for s, e in librosa.effects.split(y, top_db=30)]
    gaps = _pauses_ms(intervals)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    est_words = len(onsets) / 1.5
    return ProsodyFeatures(
        speech_rate_wpm=(est_words / duration_s) * 60.0 if duration_s > 0 else 0.0,
        pause_count=len(gaps),
        pause_mean_ms=sum(gaps) / len(gaps) if gaps else 0.0,
        pause_p90_ms=_percentile(gaps, 0.9),
        pitch_variance=pitch_variance,
        laughter_count=0,
        duration_s=duration_s,
    )
