"""SQLite persistence for engineered market features.

Note the `with closing(...)` pattern below -- this is the fix for the
`conn.close` bug (missing parentheses, so the connection was never
actually closed) from the original engine.py.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd


def save_features(df: pd.DataFrame, db_path: str, table: str) -> None:
    """Persist engineered features to a SQLite table, replacing any prior data.

    Args:
        df: DataFrame of engineered features, indexed by date.
        db_path: Filesystem path to the SQLite database.
        table: Destination table name.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        df.to_sql(table, conn, if_exists="replace", index=True, index_label="date")


def load_features(db_path: str, table: str) -> pd.DataFrame:
    """Load engineered features from a SQLite table.

    Args:
            db_path: Filesystem path to the SQLite database.
            table: Source table name.

    Returns:
            DataFrame of engineered features indexed by date (parsed as
            datetime), sorted chronologically ascending.

    Raises:
            sqlite3.OperationalError: If the table doesn't exist.
    """

    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, index_col="date", parse_dates=["date"])
    return df.sort_index()
