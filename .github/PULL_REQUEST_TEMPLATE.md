## What

<!-- One compact noun phrase in the title; here, what changed and why. -->

## Spec reference

<!-- Section of docs/technical-architecture.md or docs/product-plan.md this serves, e.g. §6 Problem 1. -->

## Verification

- [ ] `uv run pytest` passes locally
- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run mypy`
- [ ] New behaviour has a test that fails on revert

## Invariants touched

- [ ] Personas stay append-only (§1 rule 1)
- [ ] Ratings stay directed (§1 rule 2)
- [ ] `assembly_method` set on every group (§1 rule 3)
- [ ] Every bot decision writes an audit row (§9)
- [ ] No raw chat logs persist beyond TTL (§10)
- [ ] N/A

## Risks / rollout

<!-- What could break, who sees it, how to roll back. -->
