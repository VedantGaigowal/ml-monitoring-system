"""
train_model.py
--------------
Handles dataset generation, model training, and saving.
Uses a synthetic employee attrition dataset (like HR Analytics).
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR,  "training_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")

# ── Feature columns (used everywhere in the project) ──────────────────────────
NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]
TARGET               = "attrition"
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def generate_training_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic HR-attrition dataset.

    Numerical columns  : age, salary, hours_per_week, years_at_company
    Categorical columns: department, education
    Target             : attrition  (0 = stayed, 1 = left)
    """
    rng = np.random.default_rng(random_state)

    age              = rng.integers(22, 60,  n_samples).astype(int)
    salary           = rng.integers(30_000, 120_000, n_samples).astype(int)
    hours_per_week   = rng.integers(35, 60,  n_samples).astype(int)
    years_at_company = rng.integers(0,  20,  n_samples).astype(int)
    department       = rng.choice(["Engineering", "Sales", "HR", "Marketing"], n_samples)
    education        = rng.choice(["Bachelor", "Master", "PhD", "Associate"],  n_samples)

    # Simple business rule: low salary + long hours → higher attrition probability
    attrition_prob = (
        0.1
        + 0.3 * (salary           < 50_000)
        + 0.2 * (hours_per_week   > 50)
        + 0.1 * (years_at_company < 2)
        - 0.1 * (years_at_company > 10)
    )
    attrition_prob = np.clip(attrition_prob, 0.05, 0.95)
    attrition      = rng.binomial(1, attrition_prob).astype(int)

    df = pd.DataFrame({
        "age": age, "salary": salary,
        "hours_per_week": hours_per_week, "years_at_company": years_at_company,
        "department": department, "education": education,
        TARGET: attrition,
    })
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns in-place and return the frame."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        df[col] = le.fit_transform(df[col].astype(str))
    return df



def train_and_save(n_samples: int = 1000) -> dict:
    """
    Full training pipeline:
      1. Generate data  → save to data/training_data.csv
      2. Pre-process
      3. Train RandomForestClassifier
      4. Evaluate on hold-out test split
      5. Pickle model  → models/trained_model.pkl
    Returns a dict with train-time metrics.
    """
    print("📊  Generating training data …")
    df = generate_training_data(n_samples)
    df.to_csv(TRAIN_PATH, index=False)
    print(f"    Saved {len(df):,} rows → {TRAIN_PATH}")

    # Pre-process
    df_enc = preprocess(df)
    X = df_enc[ALL_FEATURES]
    y = df_enc[TARGET]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("🤖  Training RandomForestClassifier …")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅  Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾  Model saved → {MODEL_PATH}")

    return {
        "accuracy":  round(accuracy, 4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    metrics = train_and_save()
    print("\nTraining complete:", metrics)
