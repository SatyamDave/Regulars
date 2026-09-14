# Regulars infrastructure (Terraform)

Provisions the two managed pieces of the stack described in `docs/product-plan.md` §6
("BlueBubbles on existing hardware, Supabase, Railway, Claude API, Stripe"): Supabase
Postgres configuration and the Railway service that runs `ghcr.io/satyamdave/regulars`
(built by `.github/workflows/docker.yml`, deployed by `.github/workflows/deploy.yml`).

BlueBubbles, Stripe, and Anthropic are external SaaS/hardware with no Terraform-managed
infrastructure of their own here — their API keys are wired into the Railway service as
environment variables, matching `README.md`'s Configuration table and `.env.example`.

## Known limitations (read this before running anything)

This configuration was validated with a real `terraform init` + `terraform validate` +
`terraform plan` (dummy credentials, real provider binaries pulled from the public
Registry, no live Supabase/Railway account) while writing it, so the resource/attribute
names below are confirmed against the providers' actual schemas
(`terraform providers schema -json`), not guessed. What is NOT verified is behavior
against a real account — no credentials were available in this environment, so `apply`
may still hit things `validate` can't catch (permissions, quota, provider bugs).

**Supabase project creation is not automated here, on purpose.** The `supabase/supabase`
provider (v1.11.0 at validation time) is real and its `supabase_settings` resource, which
configures an *already-existing* project by `project_ref`, works as used in `main.tf`.
That provider version's schema has no `supabase_project` (create-a-project) resource,
though — project creation isn't exposed by Terraform today. So this configuration assumes
the project already exists and:

1. You create the Supabase project **manually**, once, via the Supabase dashboard or the
   `supabase` CLI (`supabase projects create`) — pick the plan/region, set the Postgres
   password, and note the **project ref** (e.g. `abcdefghijklmnop`) and the password you
   set.
2. You feed that project ref and password into this Terraform config as
   `supabase_project_ref` and `postgres_password`. `main.tf` then only:
   - manages non-destructive settings on that project via `supabase_settings`, and
   - composes `database_url` locally as
     `postgresql://postgres:<password>@db.<project_ref>.supabase.co:5432/postgres`
     (Supabase's standard direct-connection string shape).

If a future provider version makes full project creation reliable, replace step 1 with a
`supabase_project` resource — but verify its schema and destroy semantics (a project
resource with `terraform destroy` wired to it is dangerous for a production database)
against the current docs first.

**The Railway provider is community-maintained, not published by Railway.** This config
uses `terraform-community-providers/railway` (v0.6.2 at validation time). Its
`railway_project`, `railway_service`, `railway_variable`, and `railway_service_domain`
resources validated cleanly with the names used in `main.tf` — confirmed against the
provider's real schema, including two corrections an earlier draft got wrong:
`railway_service` has no `healthcheck_path`/`start_command` argument (health check and
restart policy come from the deployed source's own `railway.json`, already at the repo
root — Railway reads it directly, this Terraform config doesn't need to restate it), and
`railway_variable`/`railway_service_domain` key off `environment_id` (the project's
`default_environment.id`), not `project_id`. Still:

- This was validated against the provider's schema and a dummy `terraform plan`, not
  against a real Railway account (no credentials available here). Run `terraform plan`
  against a disposable/staging Railway account before a real production `apply`.
- It's maintained outside Railway, so re-confirm attribute names against
  https://registry.terraform.io/providers/terraform-community-providers/railway/latest/docs
  if you bump the version constraint in `versions.tf` — community providers at this size
  have renamed fields between minor versions before.
- `.github/workflows/deploy.yml` currently deploys by running `railway up` from CI (image
  already pushed to GHCR by `docker.yml`), not via Railway's native "deploy on git push"
  flow. This Terraform config provisions the *project and service shell* (so the service
  and its env vars exist and `railway up --service regulars` has somewhere to deploy to)
  — it does not change or replace the CI deploy mechanism. Do not delete `deploy.yml`'s
  `railway up` step in favor of Railway's GitHub auto-deploy without also removing the
  commented `source_repo`/`source_branch` alternative in `main.tf` or you'll have two
  deploy paths racing.

**What this does NOT do:**
- Does not create the Supabase project, database schema, or run migrations. Schema/
  migrations are `regulars/db/migrations/*.sql`, applied separately (see repo root
  `README.md` and `regulars/db/gen_ddl.py`) — out of scope for this Terraform config.
- Does not set the GitHub Actions repository/environment variables `PRODUCTION_URL` or
  `RAILWAY_SERVICE`, or the `RAILWAY_TOKEN` secret, that `deploy.yml` reads — those are
  GitHub configuration, not Railway/Supabase infrastructure. After `apply`, copy the
  `production_url` output into the repo's `production` environment variable
  `PRODUCTION_URL` by hand (Settings → Environments → production → Variables).
- Does not manage BlueBubbles (it runs on existing hardware per product-plan.md §6, not
  cloud infrastructure).

## Layout

```
infra/terraform/
├── versions.tf                              # required_providers (see honesty notes inline)
├── variables.tf                             # all inputs, secrets marked sensitive
├── main.tf                                  # Supabase settings + Railway project/service/vars
├── outputs.tf                               # database_url, railway_service_url, production_url
├── environments/
│   ├── staging.tfvars.example               # non-secret values, copy -> staging.tfvars
│   └── production.tfvars.example            # non-secret values, copy -> production.tfvars
├── .terraform.lock.hcl                      # committed: pins the exact provider versions validated above
└── README.md                                # this file
```

## Usage

```bash
cd infra/terraform

# One-time
terraform init

# Copy the example vars file for your environment and fill in the non-secret values
# (project ref, names, region). Do NOT put secrets in this file.
cp environments/staging.tfvars.example environments/staging.tfvars
$EDITOR environments/staging.tfvars

# Supply secrets via environment variables (never written to disk, never in shell history
# if your shell is configured to ignore leading-space commands):
export TF_VAR_supabase_access_token="sbp_..."
export TF_VAR_postgres_password="..."
export TF_VAR_railway_api_token="..."
export TF_VAR_stripe_api_key="sk_test_..."
export TF_VAR_stripe_webhook_secret="whsec_..."
export TF_VAR_anthropic_api_key="sk-ant-..."
export TF_VAR_bluebubbles_webhook_token="..."

terraform plan  -var-file=environments/staging.tfvars
terraform apply -var-file=environments/staging.tfvars
```

For production, repeat with `environments/production.tfvars` and production secret values
(live Stripe keys, etc.), ideally from a secrets manager (1Password CLI, AWS Secrets
Manager, Vault) piped into `TF_VAR_*` rather than typed by hand — this repo does not
prescribe one since none is currently in use elsewhere in the stack.

## Secrets

Never commit a `.tfvars` file containing real values, and never commit
`.terraform/` (provider plugin cache) or `terraform.tfstate*` (state contains resolved
sensitive values in plaintext unless you configure a remote encrypted backend, which this
config does not set up — see "Remote state" below). `infra/terraform/**/*.tfvars`,
`infra/terraform/.terraform/`, and `*.tfstate*` are covered by the repo root
`.gitignore`.

Two supported ways to supply secret variables:
1. `TF_VAR_<name>` environment variables (shown above) — preferred for CI and for not
   leaving secrets on disk.
2. An untracked `*.tfvars` file (e.g. `secrets.tfvars`, also gitignored by the
   `*.tfvars` pattern) passed with an additional `-var-file=secrets.tfvars`.

## Remote state

Not configured. By default `terraform apply` writes `terraform.tfstate` to this local
directory, which contains the plaintext values of every sensitive output/resource
attribute (including `database_url`). For anything beyond a single operator running this
by hand, configure a remote backend (e.g. an S3 bucket with encryption + a DynamoDB lock
table, or Terraform Cloud) in a `backend.tf` before the first real `apply` — deliberately
left out here since choosing a backend implies choosing a cloud account to hold it in,
which is outside this task's scope.
