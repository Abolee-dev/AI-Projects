from pathlib import Path
import sqlite3
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sqlrag.db"

def execute_sql_file(conn, path: Path):
    sql = path.read_text(encoding="utf-8")
    conn.executescript(sql)

def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    execute_sql_file(conn, ROOT / "sql" / "ddl" / "create_vulnerability_table.sql")

    csv_path = ROOT / "data" / "vulnerability_events_sample.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("vulnerability_events", conn, if_exists="append", index=False)

    execute_sql_file(conn, ROOT / "sql" / "views" / "vulnerability_aging_view.sql")

    conn.commit()
    conn.close()

    print(f"Database initialized at: {DB_PATH}")

if __name__ == "__main__":
    main()
