# Environments and the promotion flow

Regulars deploys to two Railway environments: `staging` and `production`. This document
covers how a change moves between them, what actually differs between the two, and the
checklist to run before promoting staging to production.

## Promotion flow

```
push/merge to `staging` branch --> CI --> deploy-staging.yml --> Railway "staging" env
                                                                        |
                                                          (manual: merge staging -> main)
                                                                        v
merge to `main` branch         --> CI --> deploy.yml         --> Railway "production" env
```

- **Staging deploy is automatic on the `staging` branch.** Pushing or merging a PR into
  `staging` triggers `.github/workflows/ci.yml` (its `push` trigger now includes
  `staging` alongside `main`), and on CI success `.github/workflows/deploy-staging.yml`
  deploys to the Railway `staging` environment using its own `RAILWAY_TOKEN`,
  `RAILWAY_SERVICE`, and `STAGING_URL` environment secrets/vars (all scoped to a GitHub
  Environment named `staging`, separate from `production`'s).
- **Promotion to production is a manual step**: merge (or fast-forward) `staging` into
  `main`. That merge lands as a normal push to `main`, which runs the existing
  `.github/workflows/deploy.yml` unchanged — no new gate was added to that path.

### Why this shape, and not the alternative

The task considered two options: (a) branch-mirrored — `staging` branch auto-deploys to
staging, `main` still auto-deploys to production, promotion is a branch merge; or (b)
PR-merge-to-main auto-deploys to staging first, with production behind a manual
promotion gate on top of that.

We chose (a) because:

- It reuses the exact trigger idiom `deploy.yml` already uses (`workflow_run` off `CI`,
  filtered by branch, same no-op-if-unconfigured credential check, same health-check
  loop) with a one-line branch-name change per file. Two structurally identical
  workflows are easier to reason about and audit than one workflow with two divergent
  code paths gated by environment.
- It does not touch `deploy.yml` at all, satisfying "must not remove or change existing
  production deploy behavior" by construction rather than by careful review of a shared
  file.
- It keeps `main` as "what's in production" and `staging` as "what's being tried next,"
  which matches how the person doing the promotion already thinks about git history —
  the promotion IS the merge, not a separate button.
- Option (b) would mean every PR merged to `main` deploys to staging automatically before
  anyone decided it was ready to try there, which conflates "reviewed and merged" with
  "should now be running against real Stripe/BlueBubbles/Postgres in staging."
  Branch-mirrored keeps that a deliberate act (push to `staging`).

## What's different between staging and production

| Aspect | Staging | Production |
|---|---|---|
| Railway service | separate service (`RAILWAY_SERVICE` var on the `staging` GitHub Environment, defaults to `regulars-staging`) | `regulars` (or `vars.RAILWAY_SERVICE` on `production`) |
| `RAILWAY_TOKEN` | staging-scoped token, `staging` environment secret | production-scoped token, `production` environment secret |
| Public URL | `vars.STAGING_URL` (staging GitHub Environment) | `vars.PRODUCTION_URL` (production GitHub Environment) |
| `ENVIRONMENT` | `staging` | `production` |
| `STRIPE_API_KEY` / `STRIPE_WEBHOOK_SECRET` | Stripe **test-mode** keys | Stripe **live** keys |
| `DATABASE_URL` | separate Supabase project/database (never the production database — do not point staging at prod data) | production Supabase project |
| `BLUEBUBBLES_WEBHOOK_TOKEN` | a token tied to a test/staging BlueBubbles server instance | production BlueBubbles instance |
| Infra-level isolation | not yet codified — see "IaC follow-up" below | — |

`ENVIRONMENT` is read by `Settings.from_env()` in `regulars/app.py` (new field,
`Literal["development", "staging", "production"]`, defaulting to `"development"`). It is
not consumed by any other code yet — the comment on the field in `regulars/app.py` marks
the intended hookup point (log verbosity, and a Sentry `environment` tag if/when a Sentry
integration lands; check for that branch before assuming it's merged). Nothing today
branches on it besides its own default, so adding it changes no existing behavior.

## Pre-promotion checklist

Run through this before merging `staging` into `main`:

- [ ] **Migrations applied.** Any new migration under `regulars/db/` has been applied to
      the staging Supabase database and staging has been running against it long enough
      to catch obvious issues (not just "the migration ran without error").
- [ ] **Smoke test staging.** `curl -fsS "$STAGING_URL/healthz"` returns `{"status":
      "ok"}`. Exercise the Stripe test-mode webhook path and the BlueBubbles inbound path
      against staging at least once since the last promotion.
- [ ] **Webhook endpoints point at staging while testing it.** Stripe dashboard (test
      mode) webhook endpoint URL and the BlueBubbles server's outbound webhook URL are
      pointed at `$STAGING_URL`, not production, while validating a change in staging.
- [ ] **Re-point webhook endpoints back after promotion, if applicable.** If webhook
      endpoints were temporarily pointed at staging for testing, confirm production's
      Stripe (live mode) and BlueBubbles webhook URLs still point at `$PRODUCTION_URL`
      before/after the promotion merge — the two should never both point at the same
      environment's URL.
- [ ] **CI is green on `staging`** before merging it into `main` (same CI suite runs on
      both branches per `.github/workflows/ci.yml`).
- [ ] **No production-only secrets leaked into staging config**, and vice versa — verify
      via each GitHub Environment's configured secrets/vars, not by inspecting code.

## IaC follow-up

There is currently no `infra/terraform/` directory in this repository, so there is no
Terraform-level staging/production separation to update. If/when infrastructure-as-code
is introduced (e.g. by a concurrent effort adding `infra/terraform/`), it should mirror
this same staging/production split: separate `environments/staging.tfvars` and
`environments/production.tfvars` (or `.tfvars.example` files), pointing at the same two
Railway services and two Supabase projects referenced above, rather than a single shared
tfvars file. This is tracked here as a follow-up, not implemented in this change.
