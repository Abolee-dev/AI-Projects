from app.database.db_connection import get_connection

class QueryExecutor:
    def execute(self, sql: str) -> list[dict]:
        with get_connection() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
