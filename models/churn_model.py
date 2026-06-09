"""
Train logistic regression churn model on customer features.

Label: is_churned column — sourced from generate_synthetic_data.py, never recomputed here.
Threshold: 0.35 instead of 0.5, to favour recall on the 25%-minority churn class.
"""
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.generate_synthetic_data import CHURN_CUTOFF_DATE  # noqa: F401 — reference

logger = logging.getLogger(__name__)

DATA_DIR  = _ROOT / "data"
MODEL_DIR = Path(__file__).parent

# Lowered below 0.5 to improve recall on the ~25% minority churn class
CHURN_THRESHOLD = 0.35


def _load_features(db_path: Path) -> pd.DataFrame:
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    return pd.read_sql("SELECT * FROM customer_features", engine)


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    svc_dummies = pd.get_dummies(df["preferred_service"], prefix="svc")
    X = pd.concat(
        [df[["days_since_last_visit", "visit_frequency", "avg_spend"]], svc_dummies],
        axis=1,
    )
    y = df["is_churned"].astype(int)
    return X, y


def train(db_path: Path) -> None:
    df = _load_features(db_path)
    X, y = _build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_s, y_train)
    logger.info("Model trained on %d samples", len(X_train))

    # Score entire customer base for the churn risk CSV
    X_all_s = scaler.transform(X)
    proba_all = model.predict_proba(X_all_s)[:, 1]

    output_df = df[["customer_id"]].copy()
    output_df["churn_score"]      = proba_all.round(4)
    output_df["churn_prediction"] = (proba_all >= CHURN_THRESHOLD).astype(int)

    out_path = DATA_DIR / "churn_predictions.csv"
    output_df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions → %s", len(output_df), out_path)
    print(f"Predictions saved to {out_path}")

    # Persist artifacts for evaluate.py
    joblib.dump(model,  MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, MODEL_DIR / "test_data.pkl")
    logger.info("Model artifacts saved to %s", MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = DATA_DIR / "simulation_dev.db"
    train(db_path)
