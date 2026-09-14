from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from regulars.bot.transport import (
    BlueBubblesTransport,
    TransportError,
    parse_bluebubbles_webhook,
)

PASSWORD = "s3cret"


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.responses.pop(0) if self.responses else {"status": 200, "data": {}}
        return httpx.Response(int(body.get("status", 200)), json=body)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def bb(recorder: Recorder) -> BlueBubblesTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    return BlueBubblesTransport("http://bb.local:1234/", PASSWORD, client)


def body(req: httpx.Request) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(req.content)
    return data


async def test_send_posts_text_with_password(bb: BlueBubblesTransport, recorder: Recorder) -> None:
    recorder.responses.append({"status": 200, "data": {"guid": "out-1"}})
    guid = await bb.send("iMessage;+;chat1", "Three options.")
    assert guid == "out-1"
    req = recorder.requests[0]
    assert req.method == "POST"
    assert req.url.path == "/api/v1/message/text"
    assert req.url.params["password"] == PASSWORD
    payload = body(req)
    assert payload["chatGuid"] == "iMessage;+;chat1"
    assert payload["message"] == "Three options."
    assert payload["method"] == "private-api"
    assert payload["tempGuid"]


async def test_send_dm_uses_handle_chat_guid(bb: BlueBubblesTransport, recorder: Recorder) -> None:
    await bb.send_dm("+14155550100", "hi")
    assert body(recorder.requests[0])["chatGuid"] == "iMessage;-;+14155550100"


async def test_create_group_chat_then_names_it(
    bb: BlueBubblesTransport, recorder: Recorder
) -> None:
    recorder.responses.append({"status": 200, "data": {"guid": "iMessage;+;chat99"}})
    chat_id = await bb.create_group_chat(["+1a", "+1b"], "Regulars")
    assert chat_id == "iMessage;+;chat99"
    create, name = recorder.requests
    assert create.url.path == "/api/v1/chat/new"
    assert body(create)["addresses"] == ["+1a", "+1b"]
    assert body(create)["message"]
    assert name.method == "PUT"
    assert name.url.raw_path == b"/api/v1/chat/iMessage%3B%2B%3Bchat99?password=s3cret"
    assert body(name) == {"displayName": "Regulars"}


async def test_add_and_remove_participant(bb: BlueBubblesTransport, recorder: Recorder) -> None:
    await bb.add_participant("iMessage;+;c", "+1x")
    await bb.remove_participant("iMessage;+;c", "+1x")
    add, remove = recorder.requests
    assert add.url.path.endswith("/participant/add") and body(add) == {"address": "+1x"}
    assert remove.url.path.endswith("/participant/remove") and body(remove) == {"address": "+1x"}


async def test_errors_raise_transport_error(bb: BlueBubblesTransport, recorder: Recorder) -> None:
    recorder.responses.append({"status": 500, "message": "boom"})
    with pytest.raises(TransportError):
        await bb.send("c", "x")
    recorder.responses.append({"status": 200, "data": {"guid": "x"}})
    await bb.send("c", "ok")  # recovers


# --- webhook -----------------------------------------------------------------------------------


def webhook(**data: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "guid": "in-1",
        "text": "2",
        "dateCreated": 1_757_800_000_000,
        "isFromMe": False,
        "handle": {"address": "+14155550100"},
        "chats": [{"guid": "iMessage;+;chat1"}],
        "attachments": [],
    }
    base.update(data)
    return {"type": "new-message", "data": base}


def test_parse_plain_text() -> None:
    msg = parse_bluebubbles_webhook(webhook())
    assert msg is not None
    assert msg.chat_id == "iMessage;+;chat1"
    assert msg.sender_handle == "+14155550100"
    assert msg.body == "2" and msg.message_id == "in-1" and msg.sent_at_ms == 1_757_800_000_000
    assert msg.reply_to is None and msg.tapback is None and msg.attachments == ()


def test_parse_reply() -> None:
    msg = parse_bluebubbles_webhook(webhook(threadOriginatorGuid="orig-7"))
    assert msg is not None and msg.reply_to == "orig-7"


def test_parse_tapback_add_and_remove() -> None:
    added = parse_bluebubbles_webhook(
        webhook(text="Loved “2”", associatedMessageType=2000, associatedMessageGuid="p:0/orig-7")
    )
    assert added is not None
    assert added.is_tapback and added.tapback == "❤️" and added.body == "❤️"
    assert added.reply_to == "orig-7"
    removed = parse_bluebubbles_webhook(
        webhook(associatedMessageType=3000, associatedMessageGuid="p:0/orig-7")
    )
    assert removed is None
    string_type = parse_bluebubbles_webhook(
        webhook(associatedMessageType="2003", associatedMessageGuid="bp:orig-8")
    )
    assert (
        string_type is not None and string_type.tapback == "😂" and string_type.reply_to == "orig-8"
    )


def test_parse_tapback_word_string_type() -> None:
    """BlueBubbles' ORM transformer renders associatedMessageType as a word string ("laugh",
    "-love") when the entity is hydrated through it, per BlueBubbles' own published example
    payloads; the raw poller path used for live webhooks assigns the raw int instead. Accept both.
    """
    added = parse_bluebubbles_webhook(
        webhook(
            text="Laughed at “2”",
            associatedMessageType="laugh",
            associatedMessageGuid="p:0/orig-7",
        )
    )
    assert added is not None
    assert added.is_tapback and added.tapback == "😂" and added.reply_to == "orig-7"

    removed = parse_bluebubbles_webhook(
        webhook(associatedMessageType="-love", associatedMessageGuid="p:0/orig-7")
    )
    assert removed is None

    sticker = parse_bluebubbles_webhook(webhook(associatedMessageType="sticker"))
    assert sticker is None


def test_parse_audio_attachment() -> None:
    msg = parse_bluebubbles_webhook(
        webhook(text=None, attachments=[{"guid": "att-1", "mimeType": "audio/x-caf"}]),
        base_url="http://bb.local:1234",
    )
    assert msg is not None
    assert msg.body == ""
    assert msg.has_audio()
    assert msg.attachments[0].url == "http://bb.local:1234/api/v1/attachment/att-1/download"


def test_parse_ignores_noise() -> None:
    assert parse_bluebubbles_webhook({"type": "typing-indicator", "data": {}}) is None
    assert parse_bluebubbles_webhook(webhook(isFromMe=True)) is None
    assert parse_bluebubbles_webhook(webhook(text="   ")) is None
    assert parse_bluebubbles_webhook(webhook(associatedMessageType=1000)) is None  # sticker
    assert parse_bluebubbles_webhook(webhook(chats=[])) is None
