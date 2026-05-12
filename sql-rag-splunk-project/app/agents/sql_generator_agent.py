class SQLGeneratorAgent:
    '''
    Prototype SQL generator.

    In production, replace generate() with an internal LLM call using:
    - User question
    - Dataset schema
    - Business rules
    - Few-shot examples

    This rule-based version is intentionally safe and predictable.
    '''

    def generate(self, question: str, schema_text: str) -> str:
        q = question.lower()

        base = '''
        SELECT
            application_name,
            asset_name,
            itso,
            severity,
            vul_id,
            mitigation_status,
            due_date,
            CAST(julianday('now') - julianday(due_date) AS INTEGER) AS overdue_days
        FROM vulnerability_events
        WHERE 1=1
        '''

        conditions = []

        if "critical" in q:
            conditions.append("severity = 'Critical'")
        elif "high" in q:
            conditions.append("severity = 'High'")

        if any(word in q for word in ["open", "unresolved", "breached", "overdue"]):
            conditions.append("mitigation_status != 'Closed'")

        if "30" in q:
            conditions.append("CAST(julianday('now') - julianday(due_date) AS INTEGER) > 30")

        if "top" in q and "application" in q:
            return '''
            SELECT
                application_name,
                severity,
                COUNT(*) AS open_vulnerability_count
            FROM vulnerability_events
            WHERE mitigation_status != 'Closed'
            GROUP BY application_name, severity
            ORDER BY open_vulnerability_count DESC
            LIMIT 5
            '''

        if "itso" in q or "owner" in q:
            return '''
            SELECT
                itso,
                severity,
                COUNT(*) AS open_vulnerability_count
            FROM vulnerability_events
            WHERE mitigation_status != 'Closed'
            GROUP BY itso, severity
            ORDER BY open_vulnerability_count DESC
            LIMIT 10
            '''

        if conditions:
            base += "\n AND " + "\n AND ".join(conditions)

        base += "\n ORDER BY overdue_days DESC LIMIT 100"
        return base
