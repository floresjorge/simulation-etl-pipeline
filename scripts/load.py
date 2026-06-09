"""
Load: upsert sales and customer features into SQLite.
"""
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent.parent / "sql" / "schema.sql"


def _engine(db_path: Path):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def _ensure_schema(engine) -> None:
    sql = _SCHEMA_PATH.read_text()
    with engine.connect() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()


def _upsert(engine, table: str, df: pd.DataFrame, pk: str) -> tuple[int, int]:
    """Insert new rows, update existing rows. Returns (inserted, updated)."""
    with engine.connect() as conn:
        existing_pks = {
            row[0]
            for row in conn.execute(text(f"SELECT {pk} FROM {table}")).fetchall()
        }

    new_df    = df[~df[pk].isin(existing_pks)]
    update_df = df[df[pk].isin(existing_pks)]

    if not new_df.empty:
        new_df.to_sql(table, engine, if_exists="append", index=False)

    if not update_df.empty:
        cols = [c for c in df.columns if c != pk]
        set_clause = ", ".join(f"{c} = :{c}" for c in cols)
        with engine.connect() as conn:
            for _, row in update_df.iterrows():
                conn.execute(
                    text(f"UPDATE {table} SET {set_clause} WHERE {pk} = :{pk}"),
                    row.to_dict(),
                )
            conn.commit()

    return len(new_df), len(update_df)


def load(sales_df: pd.DataFrame, customer_df: pd.DataFrame, db_path: Path) -> None:
    engine = _engine(db_path)
    _ensure_schema(engine)

    s_ins, s_upd = _upsert(engine, "sales", sales_df, "sale_id")
    logger.info("sales:             %6d inserted, %6d updated", s_ins, s_upd)

    f_ins, f_upd = _upsert(engine, "customer_features", customer_df, "customer_id")
    logger.info("customer_features: %6d inserted, %6d updated", f_ins, f_upd)
