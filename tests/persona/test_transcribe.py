import pytest

from regulars.persona.prosody import prosody_from_word_timestamps
from regulars.persona.transcribe import (
    FakeTranscriber,
    Transcriber,
    TranscriptionResult,
    WhisperAPITranscriber,
)


async def test_fake_transcriber_is_a_transcriber_and_feeds_prosody() -> None:
    result = TranscriptionResult(text="yeah so", words=[("yeah", 0.0, 0.3), ("so", 1.0, 1.2)])
    fake = FakeTranscriber(results=[result])
    assert isinstance(fake, Transcriber)
    got = await fake.transcribe("/tmp/a.m4a")
    assert got == result
    assert fake.calls[0].name == "a.m4a"
    assert prosody_from_word_timestamps(got.words).pause_count == 1


async def test_whisper_stub_not_wired() -> None:
    assert isinstance(WhisperAPITranscriber(), Transcriber)
    with pytest.raises(NotImplementedError):
        await WhisperAPITranscriber().transcribe("x.wav")
