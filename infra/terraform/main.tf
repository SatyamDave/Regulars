# Regulars infrastructure: Supabase (Postgres) settings + Railway (compute).
#
# See README.md "Known limitations" before running this against a real account. In short:
#   - The Supabase PROJECT is created manually, once, outside Terraform. This file only
#     configures settings on that existing project and composes the connection string.
#   - The Railway resources below use a community provider whose schema should be verified
#     against the current Registry docs before `terraform apply` (see versions.tf).

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

provider "supabase" {
  access_token = var.supabase_access_token
}

provider "railway" {
  token = var.railway_api_token
}

# ---------------------------------------------------------------------------
# Supabase: configure the already-created project
# ---------------------------------------------------------------------------

# `supabase_settings` manages configuration on an EXISTING project identified by
# `project_ref`. It does not create the project (see "Known limitations" above).
#
# Verified against the actual `supabase/supabase` v1.11.0 schema (via
# `terraform providers schema -json`, run in this environment): `api`, `auth`, `database`,
# `network`, `pooler`, and `storage` are each a single JSON-encoded *string* attribute (the
# raw body Supabase's Management API expects for that settings group), not nested blocks —
# an earlier draft of this file used HCL object syntax for `database`/`api`, which
# `terraform validate` correctly rejected. All of them are optional, so this resource is
# left with only `project_ref` set: it manages nothing beyond confirming the ref is valid
# and giving Terraform a place to grow into (e.g. `database = jsonencode({...})`) once a
# specific setting actually needs to be codified.
resource "supabase_settings" "regulars" {
  project_ref = var.supabase_project_ref
}

locals {
  # Standard Supabase Postgres connection string shape:
  #   postgresql://postgres:<password>@db.<project_ref>.supabase.co:5432/postgres
  # This is composed locally rather than read from a Terraform-managed resource because
  # Terraform does not create/own the database instance (see Known limitations).
  database_url = "postgresql://postgres:${var.postgres_password}@db.${var.supabase_project_ref}.supabase.co:5432/postgres"
}

# ---------------------------------------------------------------------------
# Railway: project + service running ghcr.io/satyamdave/regulars
# ---------------------------------------------------------------------------

resource "railway_project" "regulars" {
  name        = var.railway_project_name
  description = "Regulars (${var.environment}) — see docs/product-plan.md and docs/technical-architecture.md"
}

# Verified against the actual `terraform-community-providers/railway` v0.6.2 schema (via
# `terraform providers schema -json`, run in this environment). Two corrections from an
# earlier draft:
#   - `railway_service` has no `healthcheck_path` or `start_command` argument at all —
#     `terraform validate` rejected both as unsupported. Health check path/timeout and
#     restart policy are configured by the repo's own `railway.json` (already present at
#     the repo root, healthcheckPath "/healthz"), not by this provider; Railway reads that
#     file from the deployed source. `config_path` (pointing at a Railway config file)
#     conflicts with `source_image` per the schema, so it is not set here either — the
#     image-source deploy path picks up root-level `railway.json` on its own.
#   - `root_directory` also conflicts with `source_image` and has been removed.
resource "railway_service" "regulars" {
  project_id = railway_project.regulars.id
  name       = "regulars"

  # Prebuilt-image source, matching docker.yml pushing ghcr.io/satyamdave/regulars on main/v*.
  source_image = "ghcr.io/satyamdave/regulars:latest"

  # --- Alternative: GitHub-connected auto-deploy instead of image/CLI push ---
  # Remove `source_image` above and uncomment below if you want Railway to build+deploy on
  # push to `var.github_repo` instead of relying on .github/workflows/deploy.yml's
  # `railway up`. `source_repo_branch` is required whenever `source_repo` is set.
  # source_repo        = var.github_repo
  # source_repo_branch = "main"
}

# ---------------------------------------------------------------------------
# Railway: environment variables for the service
# ---------------------------------------------------------------------------
# One resource per variable rather than a single map, so a diff shows exactly which secret
# changed. `railway_variable` (verified schema) takes `environment_id`, not `project_id` —
# an earlier draft passed `project_id`, which `terraform validate` rejected since that
# attribute is computed-only (server-assigned), not settable. `environment_id` comes from
# the project's `default_environment`, since this config creates one Railway project per
# `var.environment` (staging/production) rather than multiple named environments inside a
# single project.
locals {
  railway_environment_id = railway_project.regulars.default_environment.id
}

resource "railway_variable" "database_url" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  name           = "DATABASE_URL"
  value          = local.database_url
}

resource "railway_variable" "stripe_api_key" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  name           = "STRIPE_API_KEY"
  value          = var.stripe_api_key
}

resource "railway_variable" "stripe_webhook_secret" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  name           = "STRIPE_WEBHOOK_SECRET"
  value          = var.stripe_webhook_secret
}

resource "railway_variable" "anthropic_api_key" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  name           = "ANTHROPIC_API_KEY"
  value          = var.anthropic_api_key
}

resource "railway_variable" "bluebubbles_webhook_token" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  name           = "BLUEBUBBLES_WEBHOOK_TOKEN"
  value          = var.bluebubbles_webhook_token
}

# ---------------------------------------------------------------------------
# Public URL
# ---------------------------------------------------------------------------

# Railway-generated `*.up.railway.app` domain for the service, so outputs.tf can surface a
# real URL without requiring a custom domain to be configured first. Verified schema:
# `environment_id` and `subdomain` are required (an earlier draft omitted both and passed
# `project_id`, which is computed-only); `domain` is the computed full hostname.
resource "railway_service_domain" "regulars" {
  environment_id = local.railway_environment_id
  service_id     = railway_service.regulars.id
  subdomain      = "${var.railway_project_name}-${var.environment}"
}
