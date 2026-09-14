from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection
from sqlalchemy.exc import IntegrityError

from regulars.db import repo, schema


def test_group_requires_assembly_method(conn: Connection) -> None:
    with pytest.raises(TypeError):
        repo.create_group(conn, zip_code="94110")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="assembly_method"):
        repo.create_group(conn, zip_code="94110", assembly_method="vibes")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", [None, "vibes"])
def test_database_rejects_missing_or_unknown_assembly_method(
    conn: Connection, method: str | None
) -> None:
    with pytest.raises(IntegrityError):
        conn.execute(
            schema.groups.insert().values(
                id="g", zip="94110", created_at=datetime.now(UTC), assembly_method=method
            )
        )


def test_both_arms_persist(conn: Connection) -> None:
    model = repo.create_group(conn, zip_code="94110", assembly_method="model")
    random = repo.create_group(conn, zip_code="94110", assembly_method="random")
    assert repo.get_group(conn, model.id).assembly_method == "model"
    assert repo.get_group(conn, random.id).assembly_method == "random"


def test_membership_lifecycle(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], group: repo.Group
) -> None:
    a, b, _ = users
    repo.add_membership(conn, group.id, a.id, joined_round=1)
    repo.add_membership(conn, group.id, b.id, joined_round=1)

    ended = repo.end_membership(conn, group.id, b.id, left_round=3, exit_reason="no_show_x2")
    assert (ended.left_round, ended.exit_reason) == (3, "no_show_x2")

    active = repo.memberships_for_group(conn, group.id, active_only=True)
    assert [m.user_id for m in active] == [a.id]

    with pytest.raises(repo.NotFound):
        repo.end_membership(conn, group.id, b.id, left_round=4, exit_reason="again")

    # Rejoining later is a new row keyed by joined_round; history is kept.
    repo.add_membership(conn, group.id, b.id, joined_round=5)
    assert len(repo.memberships_for_group(conn, group.id)) == 3

    with pytest.raises(IntegrityError):
        repo.add_membership(conn, group.id, b.id, joined_round=5)
