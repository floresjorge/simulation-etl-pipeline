"""
Evaluate churn model on the held-out test set.
Run after churn_model.py has trained and saved artifacts.
"""
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent

CHURN_THRESHOLD = 0.35   # must match churn_model.py


def evaluate() -> dict:
    model  = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    data   = joblib.load(MODEL_DIR / "test_data.pkl")

    X_test, y_test = data["X_test"], data["y_test"]
    X_test_s = scaler.transform(X_test)

    proba  = model.predict_proba(X_test_s)[:, 1]
    y_pred = (proba >= CHURN_THRESHOLD).astype(int)

    metrics = {
        "threshold": CHURN_THRESHOLD,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, proba), 4),
    }

    print(f"\nThreshold : {CHURN_THRESHOLD}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

    # Confusion matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Active", "Churned"])
    ax.set_yticklabels(["Active", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Simulation Churn Model")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout()
    out_png = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved → {out_png}")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate()
