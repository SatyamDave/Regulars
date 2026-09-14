"""Write the FastAPI-generated OpenAPI schema for ``regulars.app`` to ``openapi.json``.

Regenerate the checked-in snapshot with::

    uv run python scripts/export_openapi.py

``tests/test_app.py`` fails if the checked-in file drifts from a fresh ``app.openapi()`` call,
mirroring the ``gen_ddl`` vs ``regulars/db/migrations/0001_initial.sql`` pattern
(``regulars/db/gen_ddl.py``): this script is the generator, ``openapi.json`` at the repo root is
the pinned output, and the test is the drift check.

The app is built with placeholder settings (no real Stripe/BlueBubbles credentials needed) since
the schema only depends on route declarations, not on runtime secrets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regulars.app import Settings, create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "openapi.json"


def openapi_schema() -> dict[str, Any]:
    app = create_app(Settings())
    schema: dict[str, Any] = app.openapi()
    return schema


def render() -> str:
    return json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(render())


if __name__ == "__main__":
    main()
