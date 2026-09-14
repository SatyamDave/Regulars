from typing import Any

import pytest

from regulars.features.ingest import (
    MessageRow,
    message_row_from_bluebubbles,
    transcript_from_messages,
)


def row(id: int, user: str, at: int, body: str = "x", **kw: Any) -> MessageRow:
    return MessageRow(id=id, group_id="g1", user_id=user, sent_at_ms=at, body=body, **kw)


def test_char_count_defaults_to_body_length() -> None:
    assert row(1, "a", 0, body="hello").char_count == 5
    assert row(1, "a", 0, body="hello", char_count=99).char_count == 99


def test_transcript_orders_by_time_and_maps_ids_to_indices() -> None:
    rows = [
        row(30, "b", 3000, body="reply", reply_to=10),
        row(10, "a", 1000, body="first"),
        row(40, "a", 4000, body="❤️", tapback_target=30),
        row(20, "c", 2000, body="second"),
    ]
    t = transcript_from_messages(rows, ["a", "b", "c"])
    assert t.source == "real"
    assert t.metadata == {"group_id": "g1"}
    assert [x.text for x in t.turns] == ["first", "second", "reply", "❤️"]
    assert [x.sent_at_ms for x in t.turns] == [1000, 2000, 3000, 4000]
    assert t.turns[2].reply_to == 0
    assert t.turns[3].tapback_target == 2
    assert t.has_timing()


def test_ties_break_on_id() -> None:
    t = transcript_from_messages([row(2, "a", 5, body="two"), row(1, "a", 5, body="one")], ["a"])
    assert [x.text for x in t.turns] == ["one", "two"]


def test_dangling_references_dropped() -> None:
    rows = [
        row(1, "a", 1000, body="hi", reply_to=999),  # unknown reply target -> plain message
        row(2, "b", 2000, body="👍", tapback_target=998),  # unknown tapback target -> dropped
        row(3, "b", 3000, body="ok"),
    ]
    t = transcript_from_messages(rows, ["a", "b"])
    assert [x.text for x in t.turns] == ["hi", "ok"]
    assert t.turns[0].reply_to is None


def test_unknown_participant_rejected() -> None:
    with pytest.raises(ValueError):
        transcript_from_messages([row(1, "zed", 0)], ["a"])


BB_BASE: dict[str, Any] = {
    "ROWID": 7,
    "guid": "msg-7",
    "dateCreated": 1_700_000_000_000,
    "isFromMe": False,
    "handle": {"address": "+15550001"},
}


def test_bluebubbles_plain_message_and_thread_reply() -> None:
    plain = message_row_from_bluebubbles({**BB_BASE, "text": "hello"}, "g1")
    assert plain == MessageRow(
        id=7, group_id="g1", user_id="+15550001", sent_at_ms=1_700_000_000_000, body="hello"
    )
    reply = message_row_from_bluebubbles(
        {**BB_BASE, "text": "yes", "threadOriginatorGuid": "msg-3"},
        "g1",
        guid_to_id={"msg-3": 3},
    )
    assert reply is not None and reply.reply_to == 3
    unresolved = message_row_from_bluebubbles(
        {**BB_BASE, "text": "yes", "threadOriginatorGuid": "msg-3"}, "g1"
    )
    assert unresolved is not None and unresolved.reply_to is None


def test_bluebubbles_tapback_add_vs_remove() -> None:
    added = message_row_from_bluebubbles(
        {
            **BB_BASE,
            "text": "Loved “hello”",
            "associatedMessageType": 2000,
            "associatedMessageGuid": "p:0/msg-3",
        },
        "g1",
        guid_to_id={"msg-3": 3},
    )
    assert added is not None
    assert added.is_tapback and added.tapback_target == 3 and added.body == "❤️"
    assert added.reply_to is None

    removed = message_row_from_bluebubbles(
        {
            **BB_BASE,
            "text": "Removed a heart from “hello”",
            "associatedMessageType": "3000",
            "associatedMessageGuid": "p:0/msg-3",
        },
        "g1",
        guid_to_id={"msg-3": 3},
    )
    assert removed is None


def test_bluebubbles_string_type_and_bp_prefix() -> None:
    m = message_row_from_bluebubbles(
        {**BB_BASE, "associatedMessageType": "2003", "associatedMessageGuid": "bp:msg-9"},
        "g1",
        guid_to_id={"msg-9": 9},
    )
    assert m is not None and m.body == "😂" and m.tapback_target == 9


def test_bluebubbles_from_me_and_skips() -> None:
    mine = message_row_from_bluebubbles(
        {**BB_BASE, "isFromMe": True, "handle": None, "text": "me"}, "g1", self_user_id="u-self"
    )
    assert mine is not None and mine.user_id == "u-self"
    assert message_row_from_bluebubbles({**BB_BASE, "text": ""}, "g1") is None
    assert message_row_from_bluebubbles({**BB_BASE, "text": None}, "g1") is None
    assert message_row_from_bluebubbles({**BB_BASE, "associatedMessageType": 1000}, "g1") is None
    assert message_row_from_bluebubbles({**BB_BASE, "text": "hi", "associatedMessageType": 0}, "g1")
