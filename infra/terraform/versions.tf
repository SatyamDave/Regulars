# Provider pins for Regulars infrastructure.
#
# HONESTY NOTE (see infra/terraform/README.md "Known limitations" for the full picture).
# Both providers below were confirmed to exist and were actually resolved with a real
# `terraform init` + `terraform validate` + `terraform plan` (dummy credentials, real
# provider binaries and schemas, no live account) while writing this configuration —
# see .terraform.lock.hcl for the exact versions that were pulled and validated
# (supabase/supabase 1.11.0, terraform-community-providers/railway 0.6.2). What was NOT
# verified is behavior against a real Supabase/Railway account (no credentials available
# in this environment), so `terraform apply` may still surface things `validate` cannot
# catch (permission errors, plan/quota limits, provider bugs).
#
# - `supabase/supabase` is Supabase's own provider. Its `supabase_settings` resource
#   (configuring an EXISTING project, addressed by `project_ref`) validated cleanly and is
#   what this config uses. It has no `supabase_project` (create-a-project) resource in the
#   schema pulled here — project creation is not exposed by this provider version, which is
#   why main.tf assumes the project already exists (created once, manually, via the
#   Supabase dashboard or `supabase` CLI). Re-run `terraform providers schema -json` on a
#   newer version before assuming that has changed.
#
# - `terraform-community-providers/railway` is a community (not Railway-official)
#   provider. Its `railway_project`, `railway_service`, `railway_variable`, and
#   `railway_service_domain` resources validated cleanly with the attribute names used in
#   main.tf (confirmed via `terraform providers schema -json`, not guessed) — notably,
#   `railway_service` has no health-check-path argument (Railway reads that from the
#   deployed source's own `railway.json`, already present at the repo root) and
#   `railway_variable`/`railway_service_domain` key off `environment_id`, not `project_id`.
#   It is still maintained outside Railway, so re-check
#   https://registry.terraform.io/providers/terraform-community-providers/railway/latest/docs
#   before bumping the version constraint below.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    railway = {
      source  = "terraform-community-providers/railway"
      version = "~> 0.5"
    }

    supabase = {
      source  = "supabase/supabase"
      version = "~> 1.0"
    }
  }
}
