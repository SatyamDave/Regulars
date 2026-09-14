"""Real chat ingest: ``messages`` rows (§1) → :class:`Transcript`.

Two adapters live here:

* :func:`transcript_from_messages` turns stored :class:`MessageRow` rows into a real-source
  Transcript, mapping message-id references onto turn indices.
* :func:`message_row_from_bluebubbles` turns one raw BlueBubbles message payload into a
  :class:`MessageRow` (or ``None`` when the payload carries nothing we keep).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator

from regulars.core.transcript import Transcript, Turn

# BlueBubbles ``associatedMessageType``: 2000–2005 add a tapback, 3000–3005 remove one.
_TAPBACK_ADD_MIN, _TAPBACK_ADD_MAX = 2000, 2005
_TAPBACK_REMOVE_MIN, _TAPBACK_REMOVE_MAX = 3000, 3005
_TAPBACK_EMOJI: dict[int, str] = {
    2000: "❤️",  # love
    2001: "👍",  # like
    2002: "👎",  # dislike
    2003: "😂",  # laugh
    2004: "‼️",  # emphasize
    2005: "❓",  # question
}
# BlueBubbles' own ORM entity (bluebubbles-server: databases/imessage/entity/Message.ts) declares
# ``associatedMessageType`` with a column transformer that turns the DB integer into a word string
# (e.g. "laugh", "-love") whenever the entity is hydrated normally, and BlueBubbles' own published
# Postman example shows exactly that. The live-poller code path that actually feeds webhooks
# bypasses that transformer and assigns the raw integer instead (bluebubbles-server:
# databases/imessage/entity/decoders/MessageDecoder.ts). Since which one shows up on the wire
# depends on server internals we don't control, accept both.
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


class MessageRow(BaseModel):
    """One row of the §1 ``messages`` table.

    ``reply_to`` and ``tapback_target`` hold *message ids* (not turn indices); a tapback row
    stores the emoji in ``body``. ``char_count`` defaults to ``len(body)`` when omitted.
    """

    id: int
    group_id: str
    user_id: str
    sent_at_ms: int
    body: str
    reply_to: int | None = Field(default=None, description="Message id this replies to.")
    tapback_target: int | None = Field(default=None, description="Message id this reacts to.")
    char_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def _default_char_count(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("char_count") is None:
            body = data.get("body")
            if isinstance(body, str):
                return {**data, "char_count": len(body)}
        return data

    @property
    def is_tapback(self) -> bool:
        return self.tapback_target is not None


def transcript_from_messages(rows: list[MessageRow], participants: list[str]) -> Transcript:
    """Build a ``source='real'`` Transcript from stored message rows.

    Rows are ordered by ``sent_at_ms`` (then ``id`` for ties). Message-id references are mapped
    to turn indices; a ``reply_to`` pointing at a message not in ``rows`` is dropped (the turn
    becomes a plain message), and a tapback whose target is missing is dropped entirely, since
    without its target it would masquerade as a spoken turn whose text is an emoji.
    """
    ordered = sorted(rows, key=lambda r: (r.sent_at_ms, r.id))
    index_of: dict[int, int] = {}
    turns: list[Turn] = []
    for row in ordered:
        target = index_of.get(row.tapback_target) if row.tapback_target is not None else None
        if row.tapback_target is not None and target is None:
            continue  # dangling tapback
        reply_to = index_of.get(row.reply_to) if row.reply_to is not None else None
        index_of[row.id] = len(turns)
        turns.append(
            Turn(
                speaker_id=row.user_id,
                text=row.body,
                sent_at_ms=row.sent_at_ms,
                reply_to=reply_to,
                tapback_target=target,
            )
        )
    group_ids = {r.group_id for r in rows}
    metadata: dict[str, str | int | float] = {}
    if len(group_ids) == 1:
        metadata["group_id"] = group_ids.pop()
    return Transcript(participants=participants, turns=turns, source="real", metadata=metadata)


def message_row_from_bluebubbles(
    payload: Mapping[str, Any],
    group_id: str,
    *,
    guid_to_id: Mapping[str, int] | None = None,
    self_user_id: str = "me",
) -> MessageRow | None:
    """Adapt one raw BlueBubbles message (``/api/v1/message`` shape) to a :class:`MessageRow`.

    Fields read: ``originalROWID`` (id; BlueBubbles' internal SQLite column is ``ROWID``, but that
    name never appears in a serialized API/webhook payload), ``text``, ``dateCreated`` (ms epoch),
    ``handle.address`` (sender;
    ``self_user_id`` when ``isFromMe``), ``threadOriginatorGuid`` (inline reply target),
    ``associatedMessageType`` and ``associatedMessageGuid`` (tapbacks). BlueBubbles references
    other messages by GUID; ``guid_to_id`` resolves those to message ids. Unresolvable references
    become ``None`` and are dropped downstream by :func:`transcript_from_messages`.

    Returns ``None`` for tapback *removals* (types 3000–3005) and for messages with no text
    (attachments-only, typing indicators, unknown associated types).
    """
    kind = _associated_type(payload.get("associatedMessageType"))
    if kind is not None and _TAPBACK_REMOVE_MIN <= kind <= _TAPBACK_REMOVE_MAX:
        return None

    sender = payload.get("handle") or {}
    user_id = (
        self_user_id
        if payload.get("isFromMe")
        else str(sender.get("address") or payload.get("handleId") or "")
    )
    if not user_id:
        return None
    base: dict[str, Any] = {
        "id": int(payload["originalROWID"]),
        "group_id": group_id,
        "user_id": user_id,
        "sent_at_ms": int(payload["dateCreated"]),
    }

    if kind is not None:
        if not (_TAPBACK_ADD_MIN <= kind <= _TAPBACK_ADD_MAX):
            return None  # sticker, edit, or other associated message we do not model
        target = _resolve(_strip_part_prefix(payload.get("associatedMessageGuid")), guid_to_id)
        return MessageRow(**base, body=_TAPBACK_EMOJI[kind], tapback_target=target)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    reply_to = _resolve(payload.get("threadOriginatorGuid"), guid_to_id)
    return MessageRow(**base, body=text, reply_to=reply_to)


def _associated_type(raw: Any) -> int | None:
    """``associatedMessageType`` arrives as an int, a numeric string, a word string
    (BlueBubbles' own ``MessageTypeTransformer`` renders it as e.g. ``"laugh"``/``"-love"`` when
    the server hydrates it through its ORM), or 0/None for plain text."""
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


def _resolve(guid: str | None, guid_to_id: Mapping[str, int] | None) -> int | None:
    if guid is None or guid_to_id is None:
        return None
    return guid_to_id.get(guid)
