# Security

Regulars handles phone numbers, voice recordings, chat content, and payment intents.

- Report vulnerabilities privately to satyam@agentmade.ai. Do not open a public issue.
- Chat-log imports are encrypted per user at rest and purged on a 72-hour TTL; only the derived persona is retained (`docs/technical-architecture.md` §10).
- Ratings, removals, and every bot decision are auditable per user (§9).
- Secrets live in the deployment environment, never in the repo. CI runs gitleaks on every push.
