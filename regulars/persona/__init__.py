"""Voice interview → PersonaSpec; immutable versions; legibility + corrections; chat-log import."""

from regulars.persona.chatlog_import import (
    ImportedLog,
    decrypt_for_user,
    derive_from_chatlog,
    encrypt_for_user,
    new_user_key,
    purge_expired,
)
from regulars.persona.extract import PersonaExtractionError, extract_persona, prosody_hints
from regulars.persona.interview import INTERVIEW_PROMPTS, InterviewAnswer, InterviewState
from regulars.persona.legibility import apply_correction, numbered_fields, render_plain_english
from regulars.persona.prosody import (
    ProsodyFeatures,
    extract_prosody,
    prosody_from_word_timestamps,
)
from regulars.persona.transcribe import (
    FakeTranscriber,
    Transcriber,
    TranscriptionResult,
    WhisperAPITranscriber,
)
from regulars.persona.versioning import PersonaVersion, merge_update, next_version

__all__ = [
    "INTERVIEW_PROMPTS",
    "FakeTranscriber",
    "ImportedLog",
    "InterviewAnswer",
    "InterviewState",
    "PersonaExtractionError",
    "PersonaVersion",
    "ProsodyFeatures",
    "Transcriber",
    "TranscriptionResult",
    "WhisperAPITranscriber",
    "apply_correction",
    "decrypt_for_user",
    "derive_from_chatlog",
    "encrypt_for_user",
    "extract_persona",
    "extract_prosody",
    "merge_update",
    "new_user_key",
    "next_version",
    "numbered_fields",
    "prosody_from_word_timestamps",
    "prosody_hints",
    "purge_expired",
    "render_plain_english",
]
