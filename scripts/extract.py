"""
Extract: read raw sales CSV and validate schema.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA: dict[str, type] = {
    "sale_id":        str,
    "date":           str,
    "customer_id":    str,
    "dog_name":       str,
    "service":        str,
    "price_mxn":      float,
    "payment_method": str,
    "employee_id":    str,
    "notes":          object,   # nullable — mixed str/NaN
    "is_churned":     int,
}

REQUIRED_NON_NULL = [
    "sale_id", "date", "customer_id", "service",
    "price_mxn", "payment_method", "employee_id", "is_churned",
]


def extract(csv_path: Path) -> pd.DataFrame:
    logger.info("Reading %s", csv_path)

    df = pd.read_csv(
        csv_path,
        dtype={
            "sale_id":        str,
            "date":           str,
            "customer_id":    str,
            "dog_name":       str,
            "service":        str,
            "price_mxn":      float,
            "payment_method": str,
            "employee_id":    str,
            # notes: left as inferred to preserve NaN
        },
    )

    # Schema validation
    missing = set(EXPECTED_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"Schema mismatch — missing columns: {missing}")

    extra = set(df.columns) - set(EXPECTED_SCHEMA)
    if extra:
        logger.warning("Unexpected columns in source (will be kept): %s", extra)

    # Cast is_churned explicitly after read
    df["is_churned"] = df["is_churned"].astype(int)

    # Null checks on required columns
    for col in REQUIRED_NON_NULL:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(f"Required column '{col}' has {null_count} null values")

    logger.info("Extracted %d rows, %d columns", len(df), len(df.columns))
    return df
