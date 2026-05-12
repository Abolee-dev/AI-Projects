# Security Guidelines

Minimum controls for bank production:

1. Use internal LLM only.
2. Do not send raw confidential data outside the network.
3. Validate all SQL before execution.
4. Allow only SELECT for analytics.
5. Apply RBAC and row-level security.
6. Mask sensitive columns.
7. Log all prompts, SQL, user IDs, timestamps and actions.
8. Keep email/Jira actions behind human approval.
9. Use Vault or approved secret manager.
10. Add prompt-injection detection.
