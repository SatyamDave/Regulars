"""Contract tests for :func:`parse_bluebubbles_webhook` against BlueBubbles' verified API shape.

Every fixture under ``tests/fixtures/bluebubbles/`` is either a real payload captured verbatim from
BlueBubbles' own published Postman documentation, or built strictly from field names/types
confirmed against ``bluebubbles-server``'s actual source (see
``docs/bluebubbles-api-contract.md`` for the full citation of every field). None of it is a guess
presented as fact. This module is the durable regression check the original bot-runtime agent asked
for: it exists so that a future change to ``parse_bluebubbles_webhook`` (or a future BlueBubbles
server release) that breaks one of these confirmed shapes fails loudly here, rather than silently
in production.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regulars.bot.transport import parse_bluebubbles_webhook

FIXTURES = Path(__file__).parent.parent / "fixtures" / "bluebubbles"


def load(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    payload.pop("_source", None)
    return payload


def test_plain_message_matches_verified_message_response_shape() -> None:
    msg = parse_bluebubbles_webhook(load("webhook_new_message_plain.json"))
    assert msg is not None
    assert msg.chat_id == "iMessage;+;chat1234567890"
    assert msg.sender_handle == "+14155550100"
    assert msg.body == "On for Tuesday?"
    assert msg.message_id == "11111111-2222-3333-4444-555555555555"
    assert msg.sent_at_ms == 1_757_800_000_000
    assert msg.reply_to is None
    assert not msg.is_tapback


def test_inline_reply_uses_thread_originator_guid() -> None:
    msg = parse_bluebubbles_webhook(load("webhook_new_message_reply.json"))
    assert msg is not None
    assert msg.reply_to == "11111111-2222-3333-4444-555555555555"
    assert msg.body == "yes, 7pm works"


def test_tapback_add_as_raw_int_matches_live_poller_path() -> None:
    """The live poller that feeds real webhooks assigns the DB integer directly, bypassing
    BlueBubbles' own ORM transformer (confirmed in entity/decoders/MessageDecoder.ts)."""
    msg = parse_bluebubbles_webhook(load("webhook_tapback_add_int.json"))
    assert msg is not None
    assert msg.is_tapback and msg.tapback == "❤️"
    assert msg.reply_to == "11111111-2222-3333-4444-555555555555"


def test_tapback_add_as_word_string_matches_published_example() -> None:
    """BlueBubbles' own published Postman example shows associatedMessageType as a word string
    ("laugh"), produced by its ORM's MessageTypeTransformer. parse_bluebubbles_webhook must accept
    this form too, or a real tapback silently vanishes instead of degrading to a plain message."""
    msg = parse_bluebubbles_webhook(load("webhook_tapback_add_word_string.json"))
    assert msg is not None
    assert msg.is_tapback and msg.tapback == "😂"
    assert msg.reply_to == "11111111-2222-3333-4444-555555555555"


def test_tapback_remove_word_string_is_ignored() -> None:
    assert parse_bluebubbles_webhook(load("webhook_tapback_remove.json")) is None


def test_attachment_only_message_uses_verified_attachment_shape() -> None:
    msg = parse_bluebubbles_webhook(
        load("webhook_new_message_attachment.json"), base_url="http://bb.local:1234"
    )
    assert msg is not None
    assert msg.body == ""
    assert msg.has_audio()
    assert msg.attachments[0].url == "http://bb.local:1234/api/v1/attachment/att-guid-001/download"


def test_message_list_response_data_shape_uses_original_rowid_not_rowid() -> None:
    """BlueBubbles' own published example response (Chat / Get Chat Messages) confirms the id
    field is ``originalROWID``, never bare ``ROWID`` -- the bug this whole contract pass exists to
    catch. This asserts against the captured example directly, independent of our parsing code."""
    example = load("message_response_example.json")
    message = example["data"][0]
    assert "originalROWID" in message
    assert "ROWID" not in message
    assert message["originalROWID"] == 132938
    assert message["chats"][0]["guid"] == "iMessage;+;chat1234567890"
    assert message["associatedMessageType"] == "laugh"
