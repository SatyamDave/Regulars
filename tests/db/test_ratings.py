from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection
from sqlalchemy.exc import IntegrityError

from regulars.db import repo, schema


def test_ratings_are_directed(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a, b, _ = users
    ab = repo.insert_rating(
        conn, plan.id, rater_id=a.id, ratee_id=b.id, would_see_again=True, felt_connected_1_7=6
    )
    ba = repo.insert_rating(
        conn, plan.id, rater_id=b.id, ratee_id=a.id, would_see_again=False, felt_connected_1_7=2
    )
    rows = repo.ratings_for_plan(conn, plan.id)
    assert rows == [ab, ba]
    assert {(r.rater_id, r.ratee_id, r.would_see_again) for r in rows} == {
        (a.id, b.id, True),
        (b.id, a.id, False),
    }


def test_self_rating_rejected_by_api(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a = users[0]
    with pytest.raises(ValueError, match="directed"):
        repo.insert_rating(conn, plan.id, rater_id=a.id, ratee_id=a.id, would_see_again=True)
    assert repo.ratings_for_plan(conn, plan.id) == []


def test_self_rating_rejected_by_database(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a = users[0]
    with pytest.raises(IntegrityError):
        conn.execute(
            schema.ratings.insert().values(
                plan_id=plan.id,
                rater_id=a.id,
                ratee_id=a.id,
                would_see_again=True,
                submitted_at=datetime.now(UTC),
            )
        )


def test_same_direction_twice_rejected(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a, b, _ = users
    repo.insert_rating(conn, plan.id, rater_id=a.id, ratee_id=b.id, would_see_again=True)
    with pytest.raises(IntegrityError):
        repo.insert_rating(conn, plan.id, rater_id=a.id, ratee_id=b.id, would_see_again=False)


def test_felt_connected_range(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User], plan: repo.Plan
) -> None:
    a, b, _ = users
    with pytest.raises(ValueError):
        repo.insert_rating(
            conn, plan.id, rater_id=a.id, ratee_id=b.id, would_see_again=True, felt_connected_1_7=8
        )
