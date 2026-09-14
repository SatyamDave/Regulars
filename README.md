# Regulars

Simulate the conversation, then check it against what happened at dinner.

Specs live in `docs/`: [`product-plan.md`](docs/product-plan.md) and
[`technical-architecture.md`](docs/technical-architecture.md).

## Dev

```
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
