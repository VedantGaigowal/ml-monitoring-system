"""
feature_pipeline.py
-------------------
Simulates incoming production data and stores it for drift analysis.
Production data intentionally drifts to mimic real-world distribution shift.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PROD_PATH  = os.path.join(DATA_DIR, "production_data.csv")

NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]
TARGET               = "attrition"


def generate_production_data(
    n_samples: int = 300,
    drift_factor: float = 1.4,
    random_state: int = 99,
) -> pd.DataFrame:
    """
    Simulate drifted production data.

    drift_factor > 1.0  → salary compressed (lower values), hours inflated.
    This mimics a company-wide wage freeze + overwork period, causing
    distribution shift the model has never seen.
    """
    rng = np.random.default_rng(random_state)

    # Drifted numerical distributions
    age              = rng.integers(25, 55, n_samples).astype(int)
    salary           = (rng.integers(25_000, 90_000, n_samples) / drift_factor).astype(int)
    hours_per_week   = (rng.integers(40, 70, n_samples) * (drift_factor * 0.85)).astype(int)
    years_at_company = rng.integers(0, 15, n_samples).astype(int)

    # Drifted categorical distributions (Engineering over-represented in prod)
    department = rng.choice(
        ["Engineering", "Sales", "HR", "Marketing"],
        n_samples,
        p=[0.55, 0.20, 0.15, 0.10],   # training was [0.25, 0.25, 0.25, 0.25]
    )
    education = rng.choice(
        ["Bachelor", "Master", "PhD", "Associate"],
        n_samples,
        p=[0.50, 0.30, 0.10, 0.10],   # training was [0.25, 0.25, 0.25, 0.25]
    )

    # Labels (ground-truth available with a lag in real systems)
    attrition_prob = np.clip(
        0.1
        + 0.35 * (salary           < 45_000)
        + 0.25 * (hours_per_week   > 55)
        + 0.10 * (years_at_company < 2),
        0.05, 0.95,
    )
    attrition = rng.binomial(1, attrition_prob).astype(int)

    df = pd.DataFrame({
        "age": age, "salary": salary,
        "hours_per_week": hours_per_week, "years_at_company": years_at_company,
        "department": department, "education": education,
        TARGET: attrition,
    })
    return df


def capture_and_store(n_samples: int = 300, drift_factor: float = 1.4) -> pd.DataFrame:
    """
    Generate production data and persist to data/production_data.csv.
    In a real system this would be replaced by a Kafka / REST ingest layer.
    """
    print("🏭  Simulating production data …")
    df = generate_production_data(n_samples=n_samples, drift_factor=drift_factor)
    df.to_csv(PROD_PATH, index=False)
    print(f"    Saved {len(df):,} rows → {PROD_PATH}")
    return df


def load_features(path: str) -> pd.DataFrame:
    """Load a CSV feature store and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    capture_and_store()






""" This will be replaced by Rest, Kafka ingest layer in real system"""