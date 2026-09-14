# Regulars — repo conventions

Read `docs/technical-architecture.md` before touching anything. It is the spec; §-references in
code comments point at it.

## Layout

```
regulars/core/      shared contracts (Transcript, PersonaSpec, LLMClient, feature registry)
regulars/db/        schema + repositories (SQLAlchemy Core; SQLite in tests, Postgres/Supabase in prod)
regulars/persona/   voice interview → PersonaSpec; immutable versions; legibility + corrections
regulars/features/  feature extraction — ONE code path for sim and real transcripts
regulars/judge/     v1 linear judge, calibration, SRM decomposition, diagnostics
regulars/sim/       persona × persona harness, scenarios, anti-sycophancy
regulars/llm/       LLMClient providers (Anthropic, OpenAI-compatible for Centaur)
regulars/assembly/  group optimizer (max-min), exploration arm, continuity bonus
regulars/bot/       agent runtime: 5 tools, Plan object, Transport, audit rows
regulars/payments/  Stripe manual-capture deposits, credit ledger, webhooks (FastAPI app)
regulars/eval/      person-grouped CV, clustered SEs, mixed models, week-6 gate report
tests/              pytest; mirrors package layout
```

## Rules

- `regulars.core` is the only cross-package import surface. Subpackages do not import each
  other's internals; if you need a new shared type, it goes in core (and say so in your report).
- Personas are append-only versions. Ratings are directed. `assembly_method` is set on every group.
- Every LLM call goes through `LLMClient`; every test uses `FakeLLM`. No network in tests.
- Every bot decision writes an audit row.
- Feature extractors emit keys from `regulars.core.features.FEATURE_NAMES` only.
- Type-annotated, `mypy --strict` clean, `ruff` clean, tests pass: `uv run pytest && uv run ruff check . && uv run mypy`.
- Add dependencies to `pyproject.toml` one per line, alphabetical, only when needed.
