"""
Transform: clean data and engineer customer-level features.

Churn label is read directly from is_churned in the raw data.
CHURN_CUTOFF_DATE is imported for reference — we do NOT recompute
the churn flag here to avoid any inconsistency with training data.
"""
import logging
from datetime import date

import pandas as pd

from scripts.generate_synthetic_data import CHURN_CUTOFF_DATE  # noqa: F401 — reference

logger = logging.getLogger(__name__)

DATASET_END = date(2024, 6, 30)


def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Normalize service names: strip whitespace and title-case
    df["service"] = df["service"].str.strip().str.title()

    # Empty strings in notes → NaN for clean storage
    df["notes"] = df["notes"].where(
        df["notes"].notna() & (df["notes"].astype(str).str.strip() != ""),
        other=None,
    )

    # ── Customer-level aggregation ────────────────────────────────────────────
    grp = df.groupby("customer_id")

    last_visit    = grp["date"].max()
    first_visit   = grp["date"].min()
    visit_count   = grp["date"].count()
    total_revenue = grp["price_mxn"].sum()
    avg_spend     = grp["price_mxn"].mean()
    preferred_svc = grp["service"].agg(lambda s: s.value_counts().idxmax())
    is_churned    = grp["is_churned"].max().astype(int)

    # Months from first to last visit (floor at 1 to avoid division by zero)
    months_active    = ((last_visit - first_visit).dt.days / 30.0).clip(lower=1)
    visit_frequency  = (visit_count / months_active).round(2)

    days_since = (
        pd.Timestamp(DATASET_END) - last_visit
    ).dt.days.astype(int)

    customer_features = pd.DataFrame({
        "customer_id":          last_visit.index,
        "days_since_last_visit": days_since.values,
        "visit_frequency":       visit_frequency.values,
        "avg_spend":             avg_spend.values.round(2),
        "preferred_service":     preferred_svc.values,
        "total_revenue":         total_revenue.values.round(2),
        "is_churned":            is_churned.values,
    }).reset_index(drop=True)

    # Store dates as ISO strings for SQLite compatibility
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(
        "Transformed %d sales rows → %d customer feature rows (churn label from raw data)",
        len(df),
        len(customer_features),
    )
    return df, customer_features
