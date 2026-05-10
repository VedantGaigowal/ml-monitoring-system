"""
retrain_model.py
----------------
Demonstrates the FULL model lifecycle:

  Step 1 → Train original model on Period 1 (Baseline Jan-2023)
  Step 2 → Evaluate that model on all 4 periods
            Watch accuracy collapse from ~79% → ~55% as drift grows
  Step 3 → Retrain a new model on Period 4 (Recovery Mar-2024)
  Step 4 → Re-evaluate retrained model on all 4 periods
            Watch accuracy recover to ~80%+

This script is the heart of the "monitoring → retraining" loop story.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET               = "attrition"

# Paths
ORIG_MODEL_PATH    = os.path.join(MODEL_DIR, "trained_model.pkl")
RETRAIN_MODEL_PATH = os.path.join(MODEL_DIR, "retrained_model.pkl")

PERIOD_FILES = {
    "Jan-2023 (Baseline)":  "period1_baseline_jan2023.csv",
    "Jun-2023 (Stress)":    "period2_stress_jun2023.csv",
    "Dec-2023 (Crisis)":    "period3_crisis_dec2023.csv",
    "Mar-2024 (Recovery)":  "period4_recovery_mar2024.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and return encoded copy."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the given DataFrame.
    Uses 80/20 train/test split internally and prints accuracy.
    Returns the fitted model.
    """
    df_enc = preprocess(df)
    X = df_enc[ALL_FEATURES]
    y = df_enc[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"    Train-set accuracy (hold-out): {acc:.4f}")
    return clf


def evaluate_model(model, df: pd.DataFrame) -> dict:
    """
    Evaluate model against all rows in df.
    Returns dict with accuracy, precision, recall, f1.
    """
    df_enc = preprocess(df)
    X = df_enc[ALL_FEATURES]
    y = df_enc[TARGET]
    y_pred = model.predict(X)
    return {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y, y_pred,        average="weighted", zero_division=0), 4),
    }


def load_period(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run `python scripts/simulate_timeline.py` first."
        )
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main lifecycle demonstration
# ─────────────────────────────────────────────────────────────────────────────

def run_full_lifecycle():
    """
    Runs the complete model monitoring → drift → retrain lifecycle.
    Prints a side-by-side comparison table and saves both models.
    """

    # ── STEP 1: Train original model on Period 1 ─────────────────────────────
    print("\n" + "═"*60)
    print("  STEP 1: Training ORIGINAL model on Jan-2023 baseline data")
    print("═"*60)
    df_baseline = load_period("period1_baseline_jan2023.csv")
    original_model = train_model(df_baseline)
    with open(ORIG_MODEL_PATH, "wb") as f:
        pickle.dump(original_model, f)
    print(f"  ✅  Original model saved → {ORIG_MODEL_PATH}")

    # ── STEP 2: Evaluate original model on all 4 periods ─────────────────────
    print("\n" + "═"*60)
    print("  STEP 2: Evaluating ORIGINAL model across all time periods")
    print("═"*60)
    orig_results = {}
    for period_name, filename in PERIOD_FILES.items():
        df   = load_period(filename)
        orig_results[period_name] = evaluate_model(original_model, df)
        m    = orig_results[period_name]
        flag = "✅" if m["accuracy"] >= 0.75 else "⚠️ " if m["accuracy"] >= 0.60 else "❌"
        print(f"  {flag}  {period_name:<25}  Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}")

    # ── STEP 3: Retrain on Period 4 (Recovery data) ───────────────────────────
    print("\n" + "═"*60)
    print("  STEP 3: RETRAINING model on Mar-2024 recovery data")
    print("═"*60)
    df_recovery = load_period("period4_recovery_mar2024.csv")
    retrained_model = train_model(df_recovery)
    with open(RETRAIN_MODEL_PATH, "wb") as f:
        pickle.dump(retrained_model, f)
    print(f"  ✅  Retrained model saved → {RETRAIN_MODEL_PATH}")

    # ── STEP 4: Evaluate retrained model on all 4 periods ────────────────────
    print("\n" + "═"*60)
    print("  STEP 4: Evaluating RETRAINED model across all time periods")
    print("═"*60)
    retrain_results = {}
    for period_name, filename in PERIOD_FILES.items():
        df   = load_period(filename)
        retrain_results[period_name] = evaluate_model(retrained_model, df)
        m    = retrain_results[period_name]
        flag = "✅" if m["accuracy"] >= 0.75 else "⚠️ " if m["accuracy"] >= 0.60 else "❌"
        print(f"  {flag}  {period_name:<25}  Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}")

    # ── STEP 5: Side-by-side comparison table ─────────────────────────────────
    print("\n" + "═"*60)
    print("  COMPARISON: Original vs Retrained Model Accuracy")
    print("═"*60)
    print(f"  {'Period':<28} {'Original Acc':>14} {'Retrained Acc':>14} {'Δ Accuracy':>12}")
    print("  " + "─"*70)
    for period_name in PERIOD_FILES:
        orig_acc    = orig_results[period_name]["accuracy"]
        retrain_acc = retrain_results[period_name]["accuracy"]
        delta       = retrain_acc - orig_acc
        sign        = "+" if delta >= 0 else ""
        print(
            f"  {period_name:<28} {orig_acc:>14.4f} {retrain_acc:>14.4f} "
            f"{sign}{delta:>11.4f}"
        )

    # Save comparison to CSV for dashboard
    rows = []
    for period_name in PERIOD_FILES:
        rows.append({
            "period":           period_name,
            "orig_accuracy":    orig_results[period_name]["accuracy"],
            "orig_f1":          orig_results[period_name]["f1_score"],
            "retrain_accuracy": retrain_results[period_name]["accuracy"],
            "retrain_f1":       retrain_results[period_name]["f1_score"],
        })
    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(DATA_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  💾  Comparison saved → {comparison_path}")
    print("\n  ✅  Full lifecycle complete!\n")

    return orig_results, retrain_results


if __name__ == "__main__":
    run_full_lifecycle()
