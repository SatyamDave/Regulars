import pytest
from sqlalchemy import Connection, select, update
from sqlalchemy.exc import DatabaseError, IntegrityError

from regulars.core import PersonaSpec
from regulars.db import repo, schema


def _spec(*friction: str) -> PersonaSpec:
    return PersonaSpec.model_validate({"social": {"friction_points": list(friction)}})


def test_versions_append_from_one(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    a, b, _ = users
    assert repo.latest_persona_version(conn, a.id) is None

    v1 = repo.append_persona_version(conn, a.id, _spec("being talked over"), "voice_interview")
    v2 = repo.append_persona_version(conn, a.id, _spec("lateness"), "chat_behavior")
    other = repo.append_persona_version(conn, b.id, _spec(), "chatlog_import")

    assert (v1.version, v2.version, other.version) == (1, 2, 1)
    assert v1.id != v2.id

    latest = repo.latest_persona_version(conn, a.id)
    assert latest is not None
    assert latest.version == 2
    assert latest.spec.social.friction_points == ["lateness"]

    # Old versions stay reproducible byte-for-byte after a newer one lands.
    assert repo.get_persona_version(conn, v1.id).spec == _spec("being talked over")
    assert repo.get_persona_version(conn, v1.id).created_at.tzinfo is not None


def test_repo_exposes_no_persona_mutation(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    mutators = [
        name
        for name in dir(repo)
        if "persona" in name and name.startswith(("update", "set", "delete", "replace"))
    ]
    assert mutators == []

    v1 = repo.append_persona_version(conn, users[0].id, _spec(), "voice_interview")
    # Records are frozen: callers cannot even mutate the returned value.
    with pytest.raises(Exception, match="frozen"):
        v1.version = 2


def test_database_rejects_persona_update(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    v1 = repo.append_persona_version(conn, users[0].id, _spec("x"), "voice_interview")
    pv = schema.persona_versions
    with pytest.raises(DatabaseError, match="append-only"):
        conn.execute(update(pv).where(pv.c.id == v1.id).values(spec_json={"social": {}}))


def test_duplicate_version_is_a_constraint_violation(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    """A racing second append must fail loudly, never silently take the same version."""
    v1 = repo.append_persona_version(conn, users[0].id, _spec(), "voice_interview")
    with pytest.raises(IntegrityError):
        conn.execute(
            schema.persona_versions.insert().values(
                id="dup",
                user_id=v1.user_id,
                version=v1.version,
                spec_json={},
                source="voice_interview",
                created_at=v1.created_at,
            )
        )


def test_unknown_source_rejected(
    conn: Connection, users: tuple[repo.User, repo.User, repo.User]
) -> None:
    with pytest.raises(ValueError, match="source"):
        repo.append_persona_version(conn, users[0].id, _spec(), "guesswork")  # type: ignore[arg-type]
    assert conn.execute(select(schema.persona_versions)).all() == []
