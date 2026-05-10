"""
drift_detection.py
------------------
Detects feature-level distribution drift between training and production data.

Methods:
  • Kolmogorov-Smirnov (KS) test  – numerical features
  • Chi-Square test               – categorical features

Outputs a drift report DataFrame with columns:
  feature | test_type | statistic | p_value | drift_detected
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")

NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]

# p-value below this threshold → drift detected
P_VALUE_THRESHOLD = 0.05


# ── KS Test (numerical) ───────────────────────────────────────────────────────

def ks_test(train_col: pd.Series, prod_col: pd.Series) -> dict:
    """
    Two-sample KS test.
    Returns statistic (max distance between CDFs) and p-value.
    A small p-value (< threshold) means the distributions differ significantly.
    """
    stat, p_val = stats.ks_2samp(train_col.dropna(), prod_col.dropna())
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_val), 4)}


# ── Chi-Square Test (categorical) ─────────────────────────────────────────────

def chi_square_test(train_col: pd.Series, prod_col: pd.Series) -> dict:
    """
    Chi-square test of homogeneity.
    Aligns category counts from both datasets before testing.
    """
    all_cats = set(train_col.unique()) | set(prod_col.unique())

    train_counts = train_col.value_counts().reindex(all_cats, fill_value=0)
    prod_counts  = prod_col.value_counts().reindex(all_cats, fill_value=0)

    # chi2_contingency needs a 2×k contingency table
    contingency = np.array([train_counts.values, prod_counts.values])

    # Avoid degenerate case (all zeros in a row)
    if contingency.min() == 0 and contingency.sum() == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    stat, p_val, _, _ = stats.chi2_contingency(contingency)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_val), 4)}


# ── Drift Report ──────────────────────────────────────────────────────────────

def compute_drift_report(
    train_df: pd.DataFrame,
    prod_df:  pd.DataFrame,
    p_threshold: float = P_VALUE_THRESHOLD,
) -> pd.DataFrame:
    """
    Run drift tests on all features and return a summary DataFrame.

    Parameters
    ----------
    train_df    : Training reference data
    prod_df     : Production / incoming data
    p_threshold : p-value below which drift is flagged

    Returns
    -------
    DataFrame with columns:
        feature | test_type | statistic | p_value | drift_detected
    """
    records = []

    # Numerical features → KS test
    for feat in NUMERICAL_FEATURES:
        if feat not in train_df.columns or feat not in prod_df.columns:
            continue
        result = ks_test(train_df[feat], prod_df[feat])
        records.append({
            "feature":        feat,
            "test_type":      "KS Test",
            "statistic":      result["statistic"],
            "p_value":        result["p_value"],
            "drift_detected": result["p_value"] < p_threshold,
        })

    # Categorical features → Chi-Square test
    for feat in CATEGORICAL_FEATURES:
        if feat not in train_df.columns or feat not in prod_df.columns:
            continue
        result = chi_square_test(train_df[feat], prod_df[feat])
        records.append({
            "feature":        feat,
            "test_type":      "Chi-Square",
            "statistic":      result["statistic"],
            "p_value":        result["p_value"],
            "drift_detected": result["p_value"] < p_threshold,
        })

    return pd.DataFrame(records)


# ── Alert Printer ─────────────────────────────────────────────────────────────

def print_drift_alerts(report: pd.DataFrame) -> None:
    """Print console alerts for every drifted feature."""
    drifted = report[report["drift_detected"]]
    if drifted.empty:
        print("✅  No feature drift detected.")
        return
    for _, row in drifted.iterrows():
        print(
            f"\n⚠️  WARNING: Data Drift Detected\n"
            f"    Feature      : {row['feature']}\n"
            f"    Test         : {row['test_type']}\n"
            f"    Drift Score  : {row['statistic']:.4f}  (p={row['p_value']:.4f})\n"
            f"    Action       : Retrain Model"
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, "training_data.csv")
    prod_path  = os.path.join(DATA_DIR, "production_data.csv")

    train_df = pd.read_csv(train_path)
    prod_df  = pd.read_csv(prod_path)

    report = compute_drift_report(train_df, prod_df)
    print("\n── Drift Report ─────────────────────────────")
    print(report.to_string(index=False))
    print()
    print_drift_alerts(report)
