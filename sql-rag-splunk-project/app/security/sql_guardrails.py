import re
import sqlparse
from app.database.schema_registry import SchemaRegistry

BLOCKED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "replace", "attach", "detach", "pragma"
}

class SQLGuardrailError(Exception):
    pass

class SQLGuardrails:
    def __init__(self):
        self.schema_registry = SchemaRegistry()

    def validate(self, sql: str) -> str:
        if not sql or not sql.strip():
            raise SQLGuardrailError("Empty SQL query.")

        clean_sql = sql.strip().rstrip(";")
        parsed = sqlparse.parse(clean_sql)
        if not parsed:
            raise SQLGuardrailError("Invalid SQL syntax.")

        first_token = parsed[0].token_first(skip_cm=True)
        if not first_token or first_token.value.lower() != "select":
            raise SQLGuardrailError("Only SELECT queries are allowed.")

        lower_sql = clean_sql.lower()
        for keyword in BLOCKED_KEYWORDS:
            if re.search(rf"\b{keyword}\b", lower_sql):
                raise SQLGuardrailError(f"Blocked SQL keyword found: {keyword}")

        allowed_tables = self.schema_registry.allowed_tables()
        if not any(table.lower() in lower_sql for table in allowed_tables):
            raise SQLGuardrailError("SQL does not reference an allowed table.")

        if " limit " not in lower_sql:
            clean_sql += " LIMIT 100"

        return clean_sql
