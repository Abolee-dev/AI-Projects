# Architecture

This prototype uses one structured Splunk-style dataset: `vulnerability_events`.

Flow:

```text
User Question
→ Intent Router
→ Schema Registry
→ SQL Generator
→ SQL Guardrails
→ SQLite Execution
→ Summarizer
→ Optional Email Draft
```

For production, replace SQLite with PostgreSQL and replace the rule-based SQL generator with an internal LLM-powered Text-to-SQL agent.
