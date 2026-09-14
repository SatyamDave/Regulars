.PHONY: sync test lint typecheck check ddl run

sync:            ## install all deps incl. dev + audio extras
	uv sync --all-extras --dev

test:            ## run the suite with coverage
	uv run pytest --cov

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test   ## everything CI runs

ddl:             ## regenerate the Postgres migration from the SQLAlchemy metadata
	uv run python -m regulars.db.gen_ddl > regulars/db/migrations/0001_initial.sql

run:             ## local FastAPI server
	uv run uvicorn --factory regulars.app:app_from_env --reload
