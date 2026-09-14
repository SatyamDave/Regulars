"""Engine construction. SQLite in-memory for tests, ``DATABASE_URL`` (Postgres/Supabase) in prod."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.pool import StaticPool

from regulars.db.schema import metadata

DATABASE_URL_ENV = "DATABASE_URL"
SQLITE_MEMORY_URL = "sqlite+pysqlite:///:memory:"


def _normalize_url(url: str) -> str:
    # Supabase and Heroku hand out ``postgres://``; SQLAlchemy 2 only accepts ``postgresql://``.
    if url.startswith("postgres://"):
        return "postgresql://" + url.removeprefix("postgres://")
    return url


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def make_engine(url: str | None = None) -> Engine:
    """Create an engine for ``url``; when ``url`` is None read ``DATABASE_URL`` from the env."""
    if url is None:
        url = os.environ.get(DATABASE_URL_ENV)
        if not url:
            raise RuntimeError(f"{DATABASE_URL_ENV} is not set and no url was given")
    url = _normalize_url(url)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite") and ":memory:" in url:
        # One shared connection so every checkout sees the same in-memory database.
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":
        _enable_sqlite_foreign_keys(engine)
    return engine


def sqlite_memory_engine() -> Engine:
    """A fresh in-memory SQLite engine with every table (and trigger) created."""
    engine = make_engine(SQLITE_MEMORY_URL)
    metadata.create_all(engine)
    return engine
