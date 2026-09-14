"""Opt-in chat-log import (§10 Privacy).

Raw text never persists beyond the TTL. The flow is: the user shares an export → it is encrypted
at rest with a per-user key (:func:`encrypt_for_user`) as an :class:`ImportedLog` → persona
fields are derived from the plaintext (:func:`derive_from_chatlog`) → the log is purged when
``imported_at + ttl_hours`` passes (:func:`purge_expired`). What survives is the derived
:class:`PersonaSpec`, not the corpus. Import is never the gate to onboarding.

Fernet (AES-128-CBC + HMAC-SHA256, authenticated) from ``cryptography``; a wrong key or a
tampered blob raises :class:`DecryptionError` rather than returning garbage.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, field_validator

from regulars.core.llm import LLMClient, LLMMessage
from regulars.core.persona import PersonaSpec
from regulars.persona.extract import (
    PersonaDraft,
    draft_to_spec,
    ensure_negative_space,
    request_json,
    system_prompt,
)

DEFAULT_TTL_HOURS = 72
CHATLOG_CONFIDENCE = 0.5
"""Chat logs show behavior in a different register than a dinner table; rate them lower."""
MAX_CHATLOG_CHARS = 200_000
"""Roughly 50k tokens. Longer exports must be chunked by the caller, not silently truncated."""

CHATLOG_TASK = (
    "You will read an export of this person's chat with an AI assistant. Infer how they talk "
    "and what they care about from how they write to it: message length, whether they ask or "
    "instruct, what they raise unprompted, what irritates them, when they disengage. Do NOT "
    "record the topics of their questions as biography; record behavior."
)


class DecryptionError(ValueError):
    """Wrong key, or ciphertext was tampered with."""


def new_user_key() -> bytes:
    """A fresh per-user key. Store it separately from the ciphertext (KMS/secret store)."""
    return Fernet.generate_key()


def encrypt_for_user(key: bytes, blob: bytes) -> bytes:
    return Fernet(key).encrypt(blob)


def decrypt_for_user(key: bytes, ciphertext: bytes) -> bytes:
    try:
        return Fernet(key).decrypt(ciphertext)
    except InvalidToken as exc:
        raise DecryptionError("cannot decrypt imported log: wrong key or tampered data") from exc


class ImportedLog(BaseModel, frozen=True):
    user_id: str
    ciphertext: bytes
    imported_at: datetime
    ttl_hours: int = Field(default=DEFAULT_TTL_HOURS, gt=0)

    @field_validator("imported_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("imported_at must be timezone-aware")
        return v

    @property
    def expires_at(self) -> datetime:
        return self.imported_at + timedelta(hours=self.ttl_hours)

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at


def purge_expired(logs: Iterable[ImportedLog], now: datetime | None = None) -> list[ImportedLog]:
    """The logs that survive; everything expired is dropped. Pure — the caller persists."""
    moment = now or datetime.now(UTC)
    return [log for log in logs if not log.is_expired(moment)]


async def derive_from_chatlog(llm: LLMClient, plaintext: str) -> PersonaSpec:
    """Plaintext export → spec with ``sources=["chatlog_import"]``.

    Same strict-JSON / retry-once / re-ask-for-negative-space protocol as the interview. The
    plaintext is used for this call only; nothing here stores it.
    """
    if not plaintext.strip():
        raise ValueError("chat log is empty")
    if len(plaintext) > MAX_CHATLOG_CHARS:
        raise ValueError(
            f"chat log is {len(plaintext)} chars; max {MAX_CHATLOG_CHARS}. Chunk it first."
        )
    system = system_prompt(CHATLOG_TASK)
    user = f"CHAT EXPORT:\n{plaintext}"
    draft, history = await request_json(llm, system, [LLMMessage("user", user)], PersonaDraft)
    draft = await ensure_negative_space(llm, system, history, draft)
    return draft_to_spec(draft, "chatlog_import", default_confidence=CHATLOG_CONFIDENCE)
