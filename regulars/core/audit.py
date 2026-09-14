"""Shared audit-log helpers (§9, §10 of ``docs/technical-architecture.md``).

Lives in ``regulars.core`` because it is used by two subpackages that must not import each
other's internals: ``regulars.bot.router`` (the "why was I removed" DM handler) and
``regulars.ops.audit_cli`` (the operator-facing ``why-removed`` subcommand). Both work over
their own audit-row types (``regulars.bot.store.AuditEntry`` is a dataclass mirror of
``regulars.db.repo.AuditEntry``), so this module is typed against a structural
:class:`typing.Protocol` rather than either concrete type.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

MEMBER_REMOVED = "member_removed"


class AuditLike(Protocol):
    """The fields both audit-row types share and this module needs.

    Declared as read-only properties, not plain attributes: both concrete audit-row types
    (``regulars.db.repo.AuditEntry`` and its mirror ``regulars.bot.store.AuditEntry``) are frozen,
    and a plain-attribute ``Protocol`` requires a *settable* member, which a frozen dataclass or
    pydantic model never satisfies under mypy's structural check.
    """

    @property
    def decision_type(self) -> str: ...
    @property
    def reason(self) -> str: ...
    @property
    def created_at(self) -> datetime: ...
    @property
    def group_id(self) -> str | None: ...


def latest_removal[T: AuditLike](entries: Iterable[T], *, group_id: str | None = None) -> T | None:
    """The most recent ``member_removed`` row, optionally scoped to one group.

    Returns ``None`` when the user (or user+group pair) was never removed.
    """
    removals = [
        e
        for e in entries
        if e.decision_type == MEMBER_REMOVED and (group_id is None or e.group_id == group_id)
    ]
    if not removals:
        return None
    return max(removals, key=lambda e: e.created_at)


__all__ = ["MEMBER_REMOVED", "AuditLike", "latest_removal"]
