import json
from datetime import UTC, datetime, timedelta

import pytest

from regulars.core import FakeLLM
from regulars.persona.chatlog_import import (
    CHATLOG_CONFIDENCE,
    MAX_CHATLOG_CHARS,
    DecryptionError,
    ImportedLog,
    decrypt_for_user,
    derive_from_chatlog,
    encrypt_for_user,
    new_user_key,
    purge_expired,
)

T0 = datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_encrypt_decrypt_round_trip_and_key_isolation() -> None:
    k1, k2 = new_user_key(), new_user_key()
    assert k1 != k2
    blob = b"user: can you make this email less passive-aggressive\n"
    ct = encrypt_for_user(k1, blob)
    assert ct != blob and blob not in ct
    assert decrypt_for_user(k1, ct) == blob
    with pytest.raises(DecryptionError):
        decrypt_for_user(k2, ct)
    with pytest.raises(DecryptionError):
        decrypt_for_user(k1, ct[:-4] + b"AAAA")


def test_ttl_expiry() -> None:
    log = ImportedLog(user_id="u1", ciphertext=b"x", imported_at=T0)
    assert log.ttl_hours == 72
    assert log.expires_at == T0 + timedelta(hours=72)
    assert not log.is_expired(T0)
    assert not log.is_expired(T0 + timedelta(hours=71, minutes=59))
    assert log.is_expired(T0 + timedelta(hours=72))
    short = ImportedLog(user_id="u1", ciphertext=b"x", imported_at=T0, ttl_hours=1)
    assert short.is_expired(T0 + timedelta(hours=1))
    with pytest.raises(ValueError):
        ImportedLog(user_id="u1", ciphertext=b"x", imported_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        ImportedLog(user_id="u1", ciphertext=b"x", imported_at=T0, ttl_hours=0)


def test_purge_expired_is_pure() -> None:
    fresh = ImportedLog(user_id="a", ciphertext=b"1", imported_at=T0)
    stale = ImportedLog(user_id="b", ciphertext=b"2", imported_at=T0 - timedelta(hours=100))
    edge = ImportedLog(user_id="c", ciphertext=b"3", imported_at=T0 - timedelta(hours=72))
    logs = [fresh, stale, edge]
    kept = purge_expired(logs, T0)
    assert kept == [fresh]
    assert logs == [fresh, stale, edge]


async def test_derive_from_chatlog_sets_source_and_confidence() -> None:
    reply = {
        "conversational": {"avg_turn_length": "long", "question_tendency": "low"},
        "social": {"friction_points": ["being asked to clarify twice"], "drains": ["hedging"]},
    }
    llm = FakeLLM(replies=[json.dumps(reply)])
    spec = await derive_from_chatlog(llm, "user: rewrite this. no, tighter. tighter.")
    assert spec.provenance.sources == ["chatlog_import"]
    assert spec.conversational.avg_turn_length == "long"
    assert spec.has_negative_space()
    assert spec.provenance.confidence_by_field["social.drains"] == CHATLOG_CONFIDENCE
    system, messages = llm.calls[0]
    assert "chat" in system.lower() and "friction_points" in system
    assert "tighter" in messages[0].content


async def test_derive_from_chatlog_reasks_for_negative_space() -> None:
    llm = FakeLLM(
        replies=[
            json.dumps({"conversational": {"avg_turn_length": "short"}}),
            json.dumps({"friction_points": ["being corrected"], "drains": ["long threads"]}),
        ]
    )
    spec = await derive_from_chatlog(llm, "user: hi")
    assert len(llm.calls) == 2
    assert spec.social.friction_points == ["being corrected"]


async def test_derive_rejects_empty_and_oversized() -> None:
    with pytest.raises(ValueError, match="empty"):
        await derive_from_chatlog(FakeLLM(), "   ")
    with pytest.raises(ValueError, match="Chunk"):
        await derive_from_chatlog(FakeLLM(), "x" * (MAX_CHATLOG_CHARS + 1))
