"""
simulate_timeline.py
--------------------
Generates 4 snapshot datasets representing different time periods in a
company's lifecycle. Each period has a distinct business context that
changes the underlying data distribution.

Timeline:
  Period 1 – Jan 2023  : Baseline (healthy company)        → used for initial training
  Period 2 – Jun 2023  : Company stress (wage freeze)       → drift begins, model degrades
  Period 3 – Dec 2023  : Crisis peak (layoffs, overwork)    → severe drift, model fails badly
  Period 4 – Mar 2024  : Recovery (new hires, pay raises)   → drift resolved, retrain here

The story
─────────
A tech company hired a data scientist (you!) to predict employee attrition.
You trained the model in Jan 2023. By June, management froze salaries and
started pushing people harder. By December, there was a mass exodus — the
model, never having seen such conditions, was essentially guessing.
In March 2024 the company recovered: new hiring, pay bumps, better hours.
You retrain on this fresh data and accuracy bounces back.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute attrition probability from features
# ─────────────────────────────────────────────────────────────────────────────

def _attrition_prob(salary, hours, years, rng):
    prob = (
        0.08
        + 0.35 * (salary < 50_000)
        + 0.25 * (hours  > 52)
        + 0.10 * (years  < 2)
        - 0.08 * (years  > 10)
        + rng.normal(0, 0.03, len(salary))   # noise
    )
    return np.clip(prob, 0.04, 0.96)


# ─────────────────────────────────────────────────────────────────────────────
# Period generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_period_1_baseline(n=1000, seed=42) -> pd.DataFrame:
    """
    Jan 2023 – Healthy company.
    Salaries: $30k–$120k uniformly distributed.
    Hours:    35–58 per week.
    Departments evenly spread.
    Attrition rate: ~22%
    This becomes the INITIAL TRAINING DATA.
    """
    rng = np.random.default_rng(seed)
    age    = rng.integers(22, 60, n).astype(int)
    salary = rng.integers(30_000, 120_000, n).astype(int)
    hours  = rng.integers(35, 58, n).astype(int)
    years  = rng.integers(0, 20, n).astype(int)
    dept   = rng.choice(["Engineering","Sales","HR","Marketing"], n,
                         p=[0.25, 0.25, 0.25, 0.25])
    edu    = rng.choice(["Bachelor","Master","PhD","Associate"], n,
                         p=[0.25, 0.25, 0.25, 0.25])
    prob      = _attrition_prob(salary, hours, years, rng)
    attrition = rng.binomial(1, prob).astype(int)
    df = pd.DataFrame(dict(age=age, salary=salary, hours_per_week=hours,
                           years_at_company=years, department=dept,
                           education=edu, attrition=attrition))
    df["period"] = "Jan-2023 (Baseline)"
    return df


def generate_period_2_stress(n=400, seed=77) -> pd.DataFrame:
    """
    Jun 2023 – Company stress begins.
    • Salary freeze: top salaries capped, distribution shifts lower
    • Hours creeping up (overwork begins)
    • Engineering over-staffed relative to other depts
    • Attrition rate: ~35%  ← model starts under-predicting
    This is the first PRODUCTION BATCH that shows early drift.
    """
    rng = np.random.default_rng(seed)
    age    = rng.integers(24, 58, n).astype(int)
    # Salary freeze – upper end compressed to 85k
    salary = rng.integers(28_000, 85_000, n).astype(int)
    # Hours up – more overwork
    hours  = rng.integers(40, 65, n).astype(int)
    years  = rng.integers(0, 18, n).astype(int)
    dept   = rng.choice(["Engineering","Sales","HR","Marketing"], n,
                         p=[0.42, 0.25, 0.18, 0.15])
    edu    = rng.choice(["Bachelor","Master","PhD","Associate"], n,
                         p=[0.45, 0.30, 0.12, 0.13])
    prob      = _attrition_prob(salary, hours, years, rng)
    attrition = rng.binomial(1, prob).astype(int)
    df = pd.DataFrame(dict(age=age, salary=salary, hours_per_week=hours,
                           years_at_company=years, department=dept,
                           education=edu, attrition=attrition))
    df["period"] = "Jun-2023 (Stress)"
    return df


def generate_period_3_crisis(n=400, seed=101) -> pd.DataFrame:
    """
    Dec 2023 – Crisis peak.
    • Layoffs → workforce skews toward newer, lower-paid employees
    • Salaries: $22k–$72k (severely compressed)
    • Hours:    48–75 (unsustainable overwork)
    • Engineering department 65% of workforce (survivors of layoffs)
    • Attrition rate: ~58%  ← model predictions are basically noise
    This is the SEVERELY DRIFTED production batch.
    """
    rng = np.random.default_rng(seed)
    age    = rng.integers(22, 50, n).astype(int)           # younger workforce post-layoffs
    salary = rng.integers(22_000, 72_000, n).astype(int)   # severely depressed
    hours  = rng.integers(48, 75, n).astype(int)           # extreme overwork
    years  = rng.integers(0, 10, n).astype(int)            # senior people left
    dept   = rng.choice(["Engineering","Sales","HR","Marketing"], n,
                         p=[0.65, 0.18, 0.10, 0.07])       # Engineering dominates
    edu    = rng.choice(["Bachelor","Master","PhD","Associate"], n,
                         p=[0.55, 0.28, 0.07, 0.10])
    prob      = _attrition_prob(salary, hours, years, rng)
    attrition = rng.binomial(1, prob).astype(int)
    df = pd.DataFrame(dict(age=age, salary=salary, hours_per_week=hours,
                           years_at_company=years, department=dept,
                           education=edu, attrition=attrition))
    df["period"] = "Dec-2023 (Crisis)"
    return df


def generate_period_4_recovery(n=1000, seed=200) -> pd.DataFrame:
    """
    Mar 2024 – Recovery / new normal.
    • New hires bring the workforce back to a healthy distribution
    • Salaries raised: $35k–$130k (above original baseline)
    • Hours normalized: 36–55
    • Departments rebalanced
    • Attrition rate: ~18%  ← lower than ever (people want to stay)
    This becomes the RETRAINING DATA — the model trained on this
    dataset recovers high accuracy.
    """
    rng = np.random.default_rng(seed)
    age    = rng.integers(22, 62, n).astype(int)
    salary = rng.integers(35_000, 130_000, n).astype(int)   # above original baseline
    hours  = rng.integers(36, 55, n).astype(int)            # healthier hours
    years  = rng.integers(0, 22, n).astype(int)
    dept   = rng.choice(["Engineering","Sales","HR","Marketing"], n,
                         p=[0.30, 0.28, 0.22, 0.20])
    edu    = rng.choice(["Bachelor","Master","PhD","Associate"], n,
                         p=[0.28, 0.28, 0.22, 0.22])
    prob      = _attrition_prob(salary, hours, years, rng)
    attrition = rng.binomial(1, prob).astype(int)
    df = pd.DataFrame(dict(age=age, salary=salary, hours_per_week=hours,
                           years_at_company=years, department=dept,
                           education=edu, attrition=attrition))
    df["period"] = "Mar-2024 (Recovery)"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_periods() -> dict[str, pd.DataFrame]:
    """Generate all 4 period datasets, save each to data/, return as dict."""
    periods = {
        "period1_baseline_jan2023":  generate_period_1_baseline(),
        "period2_stress_jun2023":    generate_period_2_stress(),
        "period3_crisis_dec2023":    generate_period_3_crisis(),
        "period4_recovery_mar2024":  generate_period_4_recovery(),
    }

    print("\n📅  Generating multi-period timeline datasets …\n")
    print(f"  {'File':<38} {'Rows':>6}  {'Attrition%':>11}  Description")
    print("  " + "─" * 70)

    for filename, df in periods.items():
        path        = os.path.join(DATA_DIR, f"{filename}.csv")
        df_save     = df.drop(columns=["period"])   # save without the label column
        df_save.to_csv(path, index=False)
        rate = df["attrition"].mean() * 100
        desc = df["period"].iloc[0]
        print(f"  {filename+'.csv':<38} {len(df):>6}  {rate:>10.1f}%  {desc}")

    # Also save a combined file (useful for timeline charts in dashboard)
    combined = pd.concat(periods.values(), ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, "all_periods_combined.csv"), index=False)
    print(f"\n  {'all_periods_combined.csv':<38} {len(combined):>6}  (all periods)")
    print("\n✅  All period files saved to data/\n")
    return periods


if __name__ == "__main__":
    generate_all_periods()
