"""
prediction_monitoring.py
------------------------
Loads the trained model, runs inference on production data,
and tracks the prediction distribution over time.

Outputs:
  • prediction_logs.csv  – one row per production sample with timestamp
  • Console summary of prediction distribution
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

PROD_PATH  = os.path.join(DATA_DIR,  "production_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
LOG_PATH   = os.path.join(DATA_DIR,  "prediction_logs.csv")

NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET               = "attrition"


def load_model(path: str = MODEL_PATH):
    """Unpickle and return the trained model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns (same scheme as training)."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def run_predictions(
    prod_df:  pd.DataFrame,
    model,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for the production DataFrame.

    Adds columns:
      predicted_label  – 0 or 1
      predicted_prob   – probability of class 1
      timestamp        – synthetic timestamp (1 row per minute from start_time)
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=5)

    df_enc = preprocess(prod_df)
    X      = df_enc[ALL_FEATURES]

    predicted_label = model.predict(X)
    predicted_prob  = model.predict_proba(X)[:, 1]   # P(attrition=1)

    timestamps = [start_time + timedelta(minutes=i) for i in range(len(prod_df))]

    log_df = prod_df.copy()
    log_df["predicted_label"] = predicted_label
    log_df["predicted_prob"]  = np.round(predicted_prob, 4)
    log_df["timestamp"]       = timestamps

    return log_df


def save_logs(log_df: pd.DataFrame, path: str = LOG_PATH) -> None:
    """Persist prediction logs to CSV."""
    log_df.to_csv(path, index=False)
    print(f"💾  Prediction logs saved → {path}")


def summarise_predictions(log_df: pd.DataFrame) -> dict:
    """Return a simple distribution summary dict."""
    total     = len(log_df)
    n_pos     = int((log_df["predicted_label"] == 1).sum())
    n_neg     = total - n_pos
    avg_prob  = float(log_df["predicted_prob"].mean())
    summary = {
        "total_predictions": total,
        "predicted_attrition (1)": n_pos,
        "predicted_stayed   (0)": n_neg,
        "attrition_rate_%":  round(n_pos / total * 100, 2),
        "avg_confidence":    round(avg_prob, 4),
    }
    return summary


if __name__ == "__main__":
    # 1. Load artefacts
    model   = load_model()
    prod_df = pd.read_csv(PROD_PATH)

    # 2. Predict
    log_df = run_predictions(prod_df, model)

    # 3. Save
    save_logs(log_df)

    # 4. Print summary
    summary = summarise_predictions(log_df)
    print("\n── Prediction Distribution ──────────────────")
    for k, v in summary.items():
        print(f"  {k:<30}: {v}")
