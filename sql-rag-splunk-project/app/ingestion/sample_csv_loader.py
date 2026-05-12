import pandas as pd
from app.database.db_connection import get_connection

def load_vulnerability_sample(csv_path: str):
    df = pd.read_csv(csv_path)
    with get_connection() as conn:
        df.to_sql("vulnerability_events", conn, if_exists="append", index=False)
