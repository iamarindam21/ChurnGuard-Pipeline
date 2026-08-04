"""
Load layer - writes the feature table into a SQLite warehouse.
Swapping this for Postgres in production just means changing the
connection string - the to_sql/read_sql calls stay the same.
"""
import sqlite3
import pandas as pd

DB_PATH = "data/warehouse.db"


def load_client_features(df: pd.DataFrame, table_name: str = "client_features"):
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"[LOAD] wrote {len(df)} rows to {table_name}")
    finally:
        conn.close()


def read_table(table_name: str = "client_features") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()