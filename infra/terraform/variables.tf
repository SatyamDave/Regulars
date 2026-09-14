# Inputs for Regulars infrastructure. Non-secret values belong in
# environments/{staging,production}.tfvars.example (copy to a gitignored .tfvars to use them,
# or supply via TF_VAR_<name> env vars). Secrets are marked sensitive = true below and must
# never be written into a committed .tfvars file — see README.md "Secrets".

variable "environment" {
  description = "Deployment environment name (staging or production). Used to tag/name resources."
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be \"staging\" or \"production\"."
  }
}

# ---------------------------------------------------------------------------
# Provider authentication (not in the task's explicit variable list, but required
# for the providers in versions.tf to authenticate at all).
# ---------------------------------------------------------------------------

variable "supabase_access_token" {
  description = "Supabase personal access token (Account > Access Tokens) used by the supabase provider."
  type        = string
  sensitive   = true
}

variable "supabase_project_ref" {
  description = <<-EOT
    The project ref (e.g. "abcdefghijklmnop") of an ALREADY-CREATED Supabase project.
    Terraform does not create the Supabase project (see README.md "Known limitations");
    this ref is obtained once, manually, after creating the project in the Supabase
    dashboard, and is used to address `supabase_settings` and to build `database_url`.
  EOT
  type        = string
}

variable "railway_api_token" {
  description = "Railway API token (account or team token) used by the railway provider."
  type        = string
  sensitive   = true
}

# ---------------------------------------------------------------------------
# Task-specified variables
# ---------------------------------------------------------------------------

variable "supabase_project_name" {
  description = "Human-readable name of the Supabase project. Informational/tagging only, since the project itself is created manually (see Known limitations); used to sanity-check `supabase_project_ref` points at the project you expect."
  type        = string
}

variable "supabase_region" {
  description = "Supabase project region (e.g. \"us-east-1\"). Informational/tagging only for the same reason as supabase_project_name."
  type        = string
  default     = "us-east-1"
}

variable "postgres_password" {
  description = "Postgres password for the Supabase project's `postgres` role. Used only to compose database_url; Terraform does not set this password on the Supabase project (do that once via the dashboard when creating the project, or via `supabase_settings` if/when the provider supports it for your plan)."
  type        = string
  sensitive   = true
}

variable "railway_project_name" {
  description = "Name of the Railway project that hosts the Regulars service."
  type        = string
}

variable "stripe_api_key" {
  description = "Stripe secret API key (STRIPE_API_KEY). Manual-capture deposits per product-plan.md §6."
  type        = string
  sensitive   = true
}

variable "stripe_webhook_secret" {
  description = "Stripe webhook signing secret (STRIPE_WEBHOOK_SECRET) for /webhooks/stripe."
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key (ANTHROPIC_API_KEY) for scoring and cheap first-pass models."
  type        = string
  sensitive   = true
}

variable "bluebubbles_webhook_token" {
  description = "Shared token (BLUEBUBBLES_WEBHOOK_TOKEN) checked on /webhooks/bluebubbles."
  type        = string
  sensitive   = true
}

variable "github_repo" {
  description = <<-EOT
    "owner/repo" of the GitHub repository, for Railway's GitHub-connected deploy source.
    NOTE: the existing .github/workflows/deploy.yml pushes via `railway up` (Railway CLI) on a
    green CI run, not via Railway's native GitHub auto-deploy watching this repo. This variable
    is provided for a service configured to deploy on push (an alternative to the CLI flow);
    it is not required by the current deploy.yml and the default service definition in main.tf
    uses the ghcr.io image source to match today's actual deploy path exactly.
  EOT
  type        = string
  default     = "SatyamDave/Regulars"
}
