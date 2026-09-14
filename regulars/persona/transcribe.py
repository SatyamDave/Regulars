"""Speech-to-text boundary for the voice interview.

Everything downstream consumes :class:`TranscriptionResult`; the bot picks a :class:`Transcriber`
at startup and tests use :class:`FakeTranscriber`. Word timestamps are kept because
:mod:`regulars.persona.prosody` derives pacing from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from regulars.persona.prosody import WordTimestamp


class TranscriptionResult(BaseModel):
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    """``(word, start_s, end_s)`` per token; empty when the backend has no word timing."""


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(self, audio_path: str | Path) -> TranscriptionResult: ...


@dataclass
class FakeTranscriber:
    """Returns scripted results in order; the last one repeats for any further calls. No I/O."""

    results: list[TranscriptionResult] = field(default_factory=list)
    calls: list[Path] = field(default_factory=list)

    async def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        self.calls.append(Path(audio_path))
        if not self.results:
            raise RuntimeError("FakeTranscriber has no scripted results")
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class WhisperAPITranscriber:
    """Stub for OpenAI's hosted Whisper.

    Intended implementation: ``POST https://api.openai.com/v1/audio/transcriptions`` with
    ``model="whisper-1"``, ``response_format="verbose_json"`` and
    ``timestamp_granularities=["word"]``; map ``response.words[*].{word,start,end}`` onto
    :attr:`TranscriptionResult.words`. The wired client belongs in :mod:`regulars.llm` with the
    other provider clients; this stub exists so call sites can name the type today.
    """

    def __init__(self, model: str = "whisper-1") -> None:
        self.model = model

    async def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        raise NotImplementedError(
            "WhisperAPITranscriber is not wired yet; see the class docstring for the endpoint"
        )
