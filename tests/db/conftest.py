from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, Engine

from regulars.db import repo
from regulars.db.engine import sqlite_memory_engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = sqlite_memory_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    with engine.begin() as connection:
        yield connection


@pytest.fixture
def users(conn: Connection) -> tuple[repo.User, repo.User, repo.User]:
    return (
        repo.create_user(conn, phone_hash="h-a", zip_code="94110", user_id="a"),
        repo.create_user(conn, phone_hash="h-b", zip_code="94110", user_id="b"),
        repo.create_user(conn, phone_hash="h-c", zip_code="94110", user_id="c"),
    )


@pytest.fixture
def group(conn: Connection) -> repo.Group:
    return repo.create_group(conn, zip_code="94110", assembly_method="model")


@pytest.fixture
def plan(conn: Connection, group: repo.Group) -> repo.Plan:
    return repo.create_plan(conn, group.id)
