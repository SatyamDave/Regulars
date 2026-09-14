"""Schema and repositories (technical-architecture.md §1, §9).

SQLAlchemy Core only. SQLite in tests, Postgres/Supabase in prod. Repository functions take a
``Connection`` and never commit; the caller owns the transaction
(``with engine.begin() as conn: ...``).
"""

from regulars.db.engine import make_engine, sqlite_memory_engine
from regulars.db.schema import metadata

__all__ = ["make_engine", "metadata", "sqlite_memory_engine"]
