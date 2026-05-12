class SummarizerAgent:
    def summarize(self, question: str, rows: list[dict]) -> str:
        if not rows:
            return "No matching records were found for the requested criteria."

        q = question.lower()

        if "executive" in q or "summary" in q or "bluf" in q:
            return self._executive_summary(rows)

        count = len(rows)
        top_items = rows[:5]

        lines = [f"Found {count} matching record(s)."]
        for row in top_items:
            app = row.get("application_name", "N/A")
            owner = row.get("itso", "N/A")
            severity = row.get("severity", "N/A")
            overdue = row.get("overdue_days", "N/A")
            vul = row.get("vul_id", "N/A")
            lines.append(f"- {app}: {severity} {vul}, owner {owner}, overdue days {overdue}")

        return "\n".join(lines)

    def _executive_summary(self, rows: list[dict]) -> str:
        critical = sum(1 for r in rows if str(r.get("severity", "")).lower() == "critical")
        owners = sorted({r.get("itso", "N/A") for r in rows})
        apps = sorted({r.get("application_name", "N/A") for r in rows})

        return (
            f"BLUF: Current vulnerability exposure remains active, with {len(rows)} open item(s), "
            f"including {critical} critical item(s). Impact is concentrated across "
            f"{len(apps)} application(s) and {len(owners)} owner(s). Leadership attention should remain on "
            f"aged critical items, owner accountability, and closure discipline."
        )
