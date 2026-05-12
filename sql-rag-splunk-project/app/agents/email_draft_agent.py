class EmailDraftAgent:
    def draft(self, question: str, rows: list[dict]) -> str:
        if not rows:
            return "Subject: Vulnerability Follow-up\n\nNo matching vulnerability records were found."

        critical_count = sum(1 for r in rows if str(r.get("severity", "")).lower() == "critical")
        owners = sorted({r.get("itso", "N/A") for r in rows})
        apps = sorted({r.get("application_name", "N/A") for r in rows})

        details = []
        for r in rows[:10]:
            details.append(
                f"- {r.get('application_name')} | {r.get('asset_name')} | "
                f"{r.get('severity')} | {r.get('vul_id')} | Due: {r.get('due_date')} | "
                f"Overdue days: {r.get('overdue_days')}"
            )

        return f'''Subject: Action Required: Overdue Vulnerability Remediation

Hi Team,

Please review the overdue vulnerability items requiring attention.

Summary:
- Total matching open items: {len(rows)}
- Critical items: {critical_count}
- Impacted applications: {", ".join(apps)}
- Owners: {", ".join(owners)}

Key records:
{chr(10).join(details)}

Request:
Please confirm remediation plan, target closure date, and any blocker requiring escalation.

Regards,
SQL-RAG Assistant
'''
