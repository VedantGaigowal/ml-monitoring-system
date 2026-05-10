"""
performance_monitoring.py
--------------------------
Computes classification performance metrics by comparing the model's
predictions against ground-truth labels in the production dataset.

In a real system, ground-truth labels arrive with a lag (hours / days).
Here we simulate that by using the synthetic labels already in the data.

Metrics:
  • Accuracy
  • Precision  (weighted)
  • Recall     (weighted)
  • F1 Score   (weighted)

Also triggers console alerts when performance drops below thresholds.
"""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_PATH = os.path.join(DATA_DIR, "prediction_logs.csv")

# Alert thresholds – tweak to taste
THRESHOLDS = {
    "accuracy":  0.75,
    "precision": 0.70,
    "recall":    0.70,
    "f1_score":  0.70,
}


def compute_metrics(log_df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from a prediction-log DataFrame.

    Expects columns: attrition (true label) and predicted_label.
    Returns a dict of metric_name → float.
    """
    y_true = log_df["attrition"]
    y_pred = log_df["predicted_label"]

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 4),
    }
    return metrics


def get_confusion_matrix(log_df: pd.DataFrame) -> pd.DataFrame:
    """Return confusion matrix as a labelled DataFrame."""
    y_true = log_df["attrition"]
    y_pred = log_df["predicted_label"]
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual: Stayed", "Actual: Left"],
        columns=["Pred: Stayed",  "Pred: Left"],
    )


def check_performance_alerts(metrics: dict) -> list[str]:
    """
    Compare each metric against its threshold.
    Returns a list of alert strings (empty if no issues).
    """
    alerts = []
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric, 1.0)
        if value < threshold:
            alerts.append(
                f"⚠️  ALERT: {metric.upper()} dropped to {value:.4f} "
                f"(threshold = {threshold}). Consider retraining."
            )
    return alerts


def run_performance_monitoring(log_path: str = LOG_PATH) -> dict:
    """
    Full pipeline:
      1. Load prediction logs
      2. Compute metrics
      3. Print results + alerts
    Returns metrics dict.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Prediction log not found: {log_path}\n"
            "Run prediction_monitoring.py first."
        )

    log_df  = pd.read_csv(log_path)
    metrics = compute_metrics(log_df)
    alerts  = check_performance_alerts(metrics)

    print("\n── Performance Metrics ──────────────────────")
    for name, val in metrics.items():
        flag = " ⚠️" if val < THRESHOLDS.get(name, 0) else " ✅"
        print(f"  {name:<12}: {val:.4f}{flag}")

    print("\n── Confusion Matrix ─────────────────────────")
    print(get_confusion_matrix(log_df).to_string())

    if alerts:
        print("\n── Alerts ───────────────────────────────────")
        for a in alerts:
            print(f"  {a}")
    else:
        print("\n✅  All performance metrics within acceptable range.")

    return metrics


if __name__ == "__main__":
    run_performance_monitoring()
