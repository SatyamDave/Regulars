output "database_url" {
  description = "Postgres connection string for DATABASE_URL. Sensitive: contains the password."
  value       = local.database_url
  sensitive   = true
}

output "railway_service_url" {
  description = "Public HTTPS URL of the Railway-hosted Regulars service."
  value       = "https://${railway_service_domain.regulars.domain}"
}

output "production_url" {
  description = <<-EOT
    Same value as railway_service_url, named to match the `PRODUCTION_URL` repository/
    environment *variable* (not secret) that .github/workflows/deploy.yml reads as
    `vars.PRODUCTION_URL` for its environment URL and post-deploy health check. This output
    does not set that GitHub variable automatically -- copy it into the repo's "production"
    environment variables (Settings > Environments > production > Variables) after apply.
  EOT
  value       = "https://${railway_service_domain.regulars.domain}"
}
