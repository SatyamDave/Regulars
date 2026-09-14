"""Transport: the one swappable seam between the bot and iMessage (§9).

``Transport`` is the outbound interface; :func:`parse_bluebubbles_webhook` is the inbound one.
``BlueBubblesTransport`` is the only production implementation today. Along with
``regulars.features.ingest`` this is the only module allowed to know BlueBubbles payload shapes.

BlueBubbles REST assumptions (server v1.x, private API enabled). Each is one line to change:

- Auth: ``?password=<server password>`` query parameter on every request.
- Send text:            ``POST /api/v1/message/text``
                        ``{"chatGuid", "tempGuid", "message", "method": "private-api"}``
- Create chat:          ``POST /api/v1/chat/new`` ``{"addresses": [...], "message": "..."}``.
                        The server requires an initial message to create a chat; we send
                        :data:`voice.CHAT_OPENED`. Response ``data.guid`` is the chat id.
- Name a group:         ``PUT /api/v1/chat/<guid>`` ``{"displayName": "..."}``
- Add participant:      ``POST /api/v1/chat/<guid>/participant/add`` ``{"address": "..."}``
- Remove participant:   ``POST /api/v1/chat/<guid>/participant/remove`` ``{"address": "..."}``
- DM chat guid:         ``iMessage;-;<handle>``; group chat guids look like ``iMessage;+;chat…``.
- Attachment download:  ``GET /api/v1/attachment/<guid>/download``
- Webhook:              ``{"type": "new-message", "data": {<message>}}`` where the message has
                        ``guid``, ``text``, ``dateCreated`` (ms epoch), ``isFromMe``,
                        ``handle.address``, ``chats[0].guid``, ``threadOriginatorGuid`` (inline
                        reply target), ``associatedMessageType``/``associatedMessageGuid``
                        (tapbacks; 2000-2005 add, 3000-3005 remove; the type can arrive as either
                        that raw int or a word string like ``"laugh"``/``"-love"`` depending on
                        which BlueBubbles code path produced it), ``attachments[]`` with ``guid``
                        and ``mimeType``.

All of the above verified against ``bluebubbles-server`` source and BlueBubbles' own published
example payloads; see ``docs/bluebubbles-api-contract.md`` for the full per-field citation.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx

from regulars.bot import voice

AttachmentKind = Literal["audio", "image", "video", "file"]

# Kept in sync with regulars.features.ingest; not imported (subpackages do not share internals).
_TAPBACK_EMOJI: dict[int, str] = {
    2000: "❤️",
    2001: "👍",
    2002: "👎",
    2003: "😂",
    2004: "‼️",
    2005: "❓",
}
_TAPBACK_ADD = range(2000, 2006)
_TAPBACK_REMOVE = range(3000, 3006)
# BlueBubbles' ORM entity renders ``associatedMessageType`` as a word string ("laugh", "-love", ...)
# when hydrated through its own transformer, and BlueBubbles' published example payloads show
# exactly that, while the live-poller path that feeds webhooks assigns the raw integer instead
# (verified against bluebubbles-server source; see docs/bluebubbles-api-contract.md). Accept both.
_TAPBACK_TYPE_NAMES: dict[str, int] = {
    "sticker": 1000,
    "love": 2000,
    "like": 2001,
    "dislike": 2002,
    "laugh": 2003,
    "emphasize": 2004,
    "question": 2005,
    "-love": 3000,
    "-like": 3001,
    "-dislike": 3002,
    "-laugh": 3003,
    "-emphasize": 3004,
    "-question": 3005,
}


@dataclass(frozen=True)
class Attachment:
    kind: AttachmentKind
    url: str


@dataclass(frozen=True)
class InboundMessage:
    chat_id: str
    sender_handle: str
    body: str
    sent_at_ms: int
    message_id: str
    reply_to: str | None = None
    tapback: str | None = None
    attachments: tuple[Attachment, ...] = ()

    @property
    def is_tapback(self) -> bool:
        return self.tapback is not None

    def has_audio(self) -> bool:
        return any(a.kind == "audio" for a in self.attachments)


class Transport(Protocol):
    async def send(self, chat_id: str, body: str) -> str:
        """Send to an existing chat; returns the provider message id."""
        ...

    async def create_group_chat(self, handles: Sequence[str], name: str) -> str:
        """Create a named group chat; returns the chat id."""
        ...

    async def send_dm(self, handle: str, body: str) -> str: ...

    async def add_participant(self, chat_id: str, handle: str) -> None: ...

    async def remove_participant(self, chat_id: str, handle: str) -> None: ...


class TransportError(RuntimeError):
    """The provider rejected a request."""


# --- fake ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outbound:
    kind: Literal["send", "dm", "create_group", "add", "remove"]
    target: str
    body: str = ""


@dataclass
class FakeTransport:
    """Records every outbound call. ``groups`` maps chat_id -> current participant handles."""

    outbound: list[Outbound] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    _counter: int = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"msg-{self._counter}"

    async def send(self, chat_id: str, body: str) -> str:
        self.outbound.append(Outbound("send", chat_id, body))
        return self._next_id()

    async def create_group_chat(self, handles: Sequence[str], name: str) -> str:
        chat_id = f"chat-{len(self.groups) + 1}"
        self.groups[chat_id] = list(handles)
        self.outbound.append(Outbound("create_group", chat_id, name))
        return chat_id

    async def send_dm(self, handle: str, body: str) -> str:
        self.outbound.append(Outbound("dm", handle, body))
        return self._next_id()

    async def add_participant(self, chat_id: str, handle: str) -> None:
        self.groups.setdefault(chat_id, []).append(handle)
        self.outbound.append(Outbound("add", chat_id, handle))

    async def remove_participant(self, chat_id: str, handle: str) -> None:
        members = self.groups.setdefault(chat_id, [])
        if handle in members:
            members.remove(handle)
        self.outbound.append(Outbound("remove", chat_id, handle))

    # test helpers
    def sent_to(self, chat_id: str) -> list[str]:
        return [o.body for o in self.outbound if o.kind == "send" and o.target == chat_id]

    def dms_to(self, handle: str) -> list[str]:
        return [o.body for o in self.outbound if o.kind == "dm" and o.target == handle]

    def all_bodies(self) -> list[str]:
        return [o.body for o in self.outbound if o.kind in ("send", "dm")]


# --- BlueBubbles --------------------------------------------------------------------------------


def dm_chat_id(handle: str) -> str:
    return f"iMessage;-;{handle}"


class BlueBubblesTransport:
    def __init__(self, base_url: str, password: str, http: httpx.AsyncClient) -> None:
        self._base = base_url.rstrip("/")
        self._password = password
        self._http = http

    async def _call(self, method: str, path: str, json: Mapping[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        response = await self._http.request(
            method, url, params={"password": self._password}, json=json
        )
        if response.status_code >= 400:
            raise TransportError(f"{method} {path} -> HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise TransportError(f"{method} {path} -> non-JSON body") from exc
        status = payload.get("status")
        if status is not None and int(status) >= 400:
            raise TransportError(f"{method} {path} -> {payload.get('message', status)}")
        return payload.get("data")

    async def send(self, chat_id: str, body: str) -> str:
        data = await self._call(
            "POST",
            "/api/v1/message/text",
            {
                "chatGuid": chat_id,
                "tempGuid": str(uuid.uuid4()),
                "message": body,
                "method": "private-api",
            },
        )
        return str((data or {}).get("guid", ""))

    async def create_group_chat(self, handles: Sequence[str], name: str) -> str:
        data = await self._call(
            "POST", "/api/v1/chat/new", {"addresses": list(handles), "message": voice.CHAT_OPENED}
        )
        chat_id = str((data or {})["guid"])
        await self._call("PUT", f"/api/v1/chat/{quote(chat_id, safe='')}", {"displayName": name})
        return chat_id

    async def send_dm(self, handle: str, body: str) -> str:
        return await self.send(dm_chat_id(handle), body)

    async def add_participant(self, chat_id: str, handle: str) -> None:
        await self._call(
            "POST", f"/api/v1/chat/{quote(chat_id, safe='')}/participant/add", {"address": handle}
        )

    async def remove_participant(self, chat_id: str, handle: str) -> None:
        await self._call(
            "POST",
            f"/api/v1/chat/{quote(chat_id, safe='')}/participant/remove",
            {"address": handle},
        )


def parse_bluebubbles_webhook(
    payload: Mapping[str, Any], *, base_url: str = ""
) -> InboundMessage | None:
    """Turn a BlueBubbles ``new-message`` webhook into an :class:`InboundMessage`.

    Returns ``None`` for events we do not act on: other event types, our own outbound messages
    (``isFromMe``), tapback removals, and messages with neither text nor attachments.
    """
    if payload.get("type") != "new-message":
        return None
    data = payload.get("data") or {}
    if data.get("isFromMe"):
        return None
    handle = (data.get("handle") or {}).get("address")
    chats = data.get("chats") or []
    if not handle or not chats:
        return None
    chat_id = str(chats[0].get("guid", ""))
    if not chat_id:
        return None

    message_id = str(data.get("guid", ""))
    sent_at_ms = int(data.get("dateCreated") or 0)
    attachments = tuple(
        Attachment(kind=_attachment_kind(a.get("mimeType")), url=_attachment_url(base_url, a))
        for a in data.get("attachments") or []
        if a.get("guid")
    )

    kind = _int_or_none(data.get("associatedMessageType"))
    if kind is not None and kind in _TAPBACK_REMOVE:
        return None
    if kind is not None and kind in _TAPBACK_ADD:
        emoji = _TAPBACK_EMOJI[kind]
        return InboundMessage(
            chat_id=chat_id,
            sender_handle=str(handle),
            body=emoji,
            sent_at_ms=sent_at_ms,
            message_id=message_id,
            reply_to=_strip_part_prefix(data.get("associatedMessageGuid")),
            tapback=emoji,
        )
    if kind is not None:
        return None  # sticker, edit, or other associated message type

    text = data.get("text")
    body = text.strip() if isinstance(text, str) else ""
    if not body and not attachments:
        return None
    reply_to = data.get("threadOriginatorGuid")
    return InboundMessage(
        chat_id=chat_id,
        sender_handle=str(handle),
        body=body,
        sent_at_ms=sent_at_ms,
        message_id=message_id,
        reply_to=str(reply_to) if reply_to else None,
        attachments=attachments,
    )


def _attachment_kind(mime: Any) -> AttachmentKind:
    if isinstance(mime, str):
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
    return "file"


def _attachment_url(base_url: str, attachment: Mapping[str, Any]) -> str:
    guid = quote(str(attachment["guid"]), safe="")
    return f"{base_url.rstrip('/')}/api/v1/attachment/{guid}/download"


def _int_or_none(raw: Any) -> int | None:
    """``associatedMessageType`` arrives as an int, a numeric string, a word string (see
    ``_TAPBACK_TYPE_NAMES``), or 0/None."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw in _TAPBACK_TYPE_NAMES:
        return _TAPBACK_TYPE_NAMES[raw]
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value or None


def _strip_part_prefix(guid: Any) -> str | None:
    """``associatedMessageGuid`` looks like ``p:0/<guid>`` or ``bp:<guid>``; keep the guid."""
    if not isinstance(guid, str):
        return None
    if "/" in guid:
        return guid.rsplit("/", 1)[1]
    if guid.startswith("bp:"):
        return guid[3:]
    return guid
