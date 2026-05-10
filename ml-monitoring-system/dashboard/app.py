"""
dashboard/app.py
----------------
Streamlit monitoring dashboard with 6 pages:
  1. Model Overview
  2. Feature Drift
  3. Prediction Monitoring
  4. Performance Monitoring
  5. Alerts
  6. Timeline & Retraining  ← NEW: shows drift over time + retrain recovery
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ── Path setup (allow imports from scripts/) ──────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR  = os.path.join(BASE_DIR, "scripts")
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "models")

sys.path.insert(0, SCRIPTS_DIR)

from drift_detection       import compute_drift_report, P_VALUE_THRESHOLD
from performance_monitoring import compute_metrics, check_performance_alerts, get_confusion_matrix

TRAIN_PATH      = os.path.join(DATA_DIR,  "training_data.csv")
PROD_PATH       = os.path.join(DATA_DIR,  "production_data.csv")
LOG_PATH        = os.path.join(DATA_DIR,  "prediction_logs.csv")
MODEL_PATH      = os.path.join(MODEL_DIR, "trained_model.pkl")
RETRAIN_PATH    = os.path.join(MODEL_DIR, "retrained_model.pkl")
COMPARISON_PATH = os.path.join(DATA_DIR,  "model_comparison.csv")
COMBINED_PATH   = os.path.join(DATA_DIR,  "all_periods_combined.csv")

PERIOD_FILES = {
    "Jan-2023 (Baseline)": "period1_baseline_jan2023.csv",
    "Jun-2023 (Stress)":   "period2_stress_jun2023.csv",
    "Dec-2023 (Crisis)":   "period3_crisis_dec2023.csv",
    "Mar-2024 (Recovery)": "period4_recovery_mar2024.csv",
}

NUMERICAL_FEATURES   = ["age", "salary", "hours_per_week", "years_at_company"]
CATEGORICAL_FEATURES = ["department", "education"]
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET               = "attrition"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    prod_df  = pd.read_csv(PROD_PATH)
    log_df   = pd.read_csv(LOG_PATH)  if os.path.exists(LOG_PATH)  else pd.DataFrame()
    return train_df, prod_df, log_df


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_training_accuracy(model, train_df):
    df_enc = preprocess(train_df)
    X      = df_enc[ALL_FEATURES]
    y      = df_enc[TARGET]
    return round(accuracy_score(y, model.predict(X)), 4)


# ── Streamlit page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("ML Monitoring")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Model Overview",
        "📉 Feature Drift",
        "🎯 Prediction Monitoring",
        "📈 Performance Monitoring",
        "🚨 Alerts",
        "⏳ Timeline & Retraining",
    ],
)

# Check data exists; prompt user if not
if not os.path.exists(TRAIN_PATH) or not os.path.exists(PROD_PATH):
    st.error(
        "⚠️  Data files not found. "
        "Run `python scripts/train_model.py` and `python scripts/feature_pipeline.py` first."
    )
    st.stop()

train_df, prod_df, log_df = load_data()
model                     = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Model Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Model Overview":
    st.title("🏠 Model Overview")
    st.markdown("High-level snapshot of the deployed classification model.")

    train_acc = get_training_accuracy(model, train_df)

    # Compute drift to decide model status
    report        = compute_drift_report(train_df, prod_df)
    any_drift     = report["drift_detected"].any()
    model_status  = "⚠️ Needs Attention" if any_drift else "✅ Healthy"
    status_colour = "warning" if any_drift else "success"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type",        "Random Forest Classifier")
    col2.metric("Training Accuracy", f"{train_acc * 100:.2f}%")
    col3.metric("Model Status",      model_status)

    st.markdown("---")
    st.subheader("Dataset Summary")
    c1, c2 = st.columns(2)
    c1.dataframe(
        pd.DataFrame({
            "Split":   ["Training", "Production"],
            "Samples": [len(train_df), len(prod_df)],
            "Features": [len(ALL_FEATURES), len(ALL_FEATURES)],
        }),
        use_container_width=True,
    )
    c2.dataframe(
        pd.DataFrame({
            "Feature Type": ["Numerical", "Categorical"],
            "Count":        [len(NUMERICAL_FEATURES), len(CATEGORICAL_FEATURES)],
            "Names":        [", ".join(NUMERICAL_FEATURES), ", ".join(CATEGORICAL_FEATURES)],
        }),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Feature Importances")
    importances = pd.Series(
        model.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances (Random Forest)")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Feature Drift
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📉 Feature Drift":
    st.title("📉 Feature Drift Detection")
    st.markdown(
        "Comparing **training** vs **production** distributions using "
        "KS Test (numerical) and Chi-Square Test (categorical)."
    )

    report = compute_drift_report(train_df, prod_df)

    # Summary bar at top
    n_drift = report["drift_detected"].sum()
    n_ok    = len(report) - n_drift
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Features Tested", len(report))
    c2.metric("Drifted Features 🔴",   int(n_drift))
    c3.metric("Stable Features 🟢",    int(n_ok))

    st.markdown("---")

    # Styled table
    def highlight_drift(row):
        colour = "#ffcccc" if row["drift_detected"] else "#ccffcc"
        return [f"background-color: {colour}"] * len(row)

    display_df = report.copy()
    display_df["drift_detected"] = display_df["drift_detected"].map({True: "🔴 DRIFT", False: "🟢 OK"})
    st.dataframe(display_df.style.apply(highlight_drift, axis=1), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribution Comparison – Numerical Features")

    cols = st.columns(2)
    for i, feat in enumerate(NUMERICAL_FEATURES):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(train_df[feat], bins=25, alpha=0.6, label="Train",      color="#4C72B0")
            ax.hist(prod_df[feat],  bins=25, alpha=0.6, label="Production", color="#DD8452")
            ax.set_title(feat)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Distribution Comparison – Categorical Features")
    cols2 = st.columns(2)
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        with cols2[i % 2]:
            train_counts = train_df[feat].value_counts(normalize=True)
            prod_counts  = prod_df[feat].value_counts(normalize=True)
            all_cats = list(set(train_counts.index) | set(prod_counts.index))
            x = np.arange(len(all_cats))

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(x - 0.2, [train_counts.get(c, 0) for c in all_cats],
                   width=0.35, label="Train",      color="#4C72B0", alpha=0.8)
            ax.bar(x + 0.2, [prod_counts.get(c, 0)  for c in all_cats],
                   width=0.35, label="Production", color="#DD8452", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=20, fontsize=8)
            ax.set_title(feat)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Prediction Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Prediction Monitoring":
    st.title("🎯 Prediction Monitoring")

    if log_df.empty:
        st.warning("No prediction logs found. Run `prediction_monitoring.py` first.")
        st.stop()

    # KPIs
    n_total = len(log_df)
    n_pos   = int((log_df["predicted_label"] == 1).sum())
    n_neg   = n_total - n_pos
    avg_p   = float(log_df["predicted_prob"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions",    n_total)
    c2.metric("Predicted Attrition",  n_pos)
    c3.metric("Predicted Stayed",     n_neg)
    c4.metric("Avg Confidence",       f"{avg_p:.3f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Pie chart – class distribution
    with col_a:
        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [n_pos, n_neg],
            labels=["Attrition (1)", "Stayed (0)"],
            autopct="%1.1f%%",
            colors=["#DD8452", "#4C72B0"],
            startangle=90,
        )
        ax.set_title("Predicted Class Split")
        st.pyplot(fig)
        plt.close(fig)

    # Histogram – confidence scores
    with col_b:
        st.subheader("Confidence Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(log_df["predicted_prob"], bins=20, color="#55A868", edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", label="Decision boundary")
        ax.set_xlabel("P(Attrition = 1)")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Probability Histogram")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Time-series rolling attrition rate
    st.markdown("---")
    st.subheader("Prediction Rate Over Time")
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    log_df_sorted = log_df.sort_values("timestamp")
    log_df_sorted["rolling_rate"] = (
        log_df_sorted["predicted_label"].rolling(window=30, min_periods=1).mean()
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(log_df_sorted["timestamp"], log_df_sorted["rolling_rate"],
            color="#4C72B0", linewidth=1.5, label="30-sample rolling attrition rate")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="50% baseline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Attrition Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Recent Predictions (last 20)")
    st.dataframe(log_df.tail(20)[
        ["timestamp", "age", "salary", "department", "predicted_label", "predicted_prob"]
    ], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Performance Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performance Monitoring":
    st.title("📈 Performance Monitoring")

    if log_df.empty:
        st.warning("No prediction logs found. Run `prediction_monitoring.py` first.")
        st.stop()

    metrics = compute_metrics(log_df)
    alerts  = check_performance_alerts(metrics)

    THRESHOLDS = {"accuracy": 0.75, "precision": 0.70, "recall": 0.70, "f1_score": 0.70}

    # KPI cards
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        delta_colour = "normal" if val >= THRESHOLDS[name] else "inverse"
        col.metric(
            label=name.replace("_", " ").title(),
            value=f"{val:.4f}",
            delta=f"threshold {THRESHOLDS[name]}",
            delta_color=delta_colour,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Bar chart of metrics
    with col_a:
        st.subheader("Metrics Bar Chart")
        fig, ax = plt.subplots(figsize=(5, 4))
        names  = list(metrics.keys())
        values = list(metrics.values())
        thresh = [THRESHOLDS[n] for n in names]
        x = np.arange(len(names))
        ax.bar(x, values, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"], alpha=0.85)
        ax.plot(x, thresh, "r--o", linewidth=1.5, markersize=5, label="Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names])
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Performance vs Thresholds")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Confusion matrix heatmap
    with col_b:
        st.subheader("Confusion Matrix")
        cm_df = get_confusion_matrix(log_df)
        cm    = cm_df.values
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]);  ax.set_xticklabels(["Pred: Stayed", "Pred: Left"])
        ax.set_yticks([0, 1]);  ax.set_yticklabels(["Actual: Stayed", "Actual: Left"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=14)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

    if alerts:
        st.markdown("---")
        for a in alerts:
            st.error(a)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Alerts
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Alerts":
    st.title("🚨 Alerts & Recommendations")

    report  = compute_drift_report(train_df, prod_df)
    metrics = compute_metrics(log_df) if not log_df.empty else {}
    perf_alerts = check_performance_alerts(metrics) if metrics else []

    # ── Drift Alerts ──────────────────────────────────────────────────────────
    st.subheader("Feature Drift Alerts")
    drift_alerts = report[report["drift_detected"]]
    if drift_alerts.empty:
        st.success("✅  No drift detected across all features.")
    else:
        for _, row in drift_alerts.iterrows():
            st.warning(
                f"⚠️  **Data Drift Detected**  \n"
                f"**Feature**: `{row['feature']}`  \n"
                f"**Test**: {row['test_type']}  \n"
                f"**Drift Score**: {row['statistic']:.4f}  (p-value = {row['p_value']:.4f})  \n"
                f"**Recommended Action**: Retrain model with recent data."
            )

    st.markdown("---")

    # ── Performance Alerts ────────────────────────────────────────────────────
    st.subheader("Performance Alerts")
    if not perf_alerts:
        st.success("✅  All performance metrics are within acceptable thresholds.")
    else:
        for a in perf_alerts:
            st.error(a)

    st.markdown("---")

    # ── Summary Table ─────────────────────────────────────────────────────────
    st.subheader("Full Drift Report")
    display_report = report.copy()
    display_report["drift_detected"] = display_report["drift_detected"].map(
        {True: "🔴 DRIFT", False: "🟢 OK"}
    )
    st.dataframe(display_report, use_container_width=True)

    if metrics:
        st.subheader("Current Performance Snapshot")
        st.table(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Timeline & Retraining
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⏳ Timeline & Retraining":
    st.title("⏳ Timeline & Retraining Story")
    st.markdown("""
    This page shows the **full model lifecycle** across 4 time periods.
    Watch the model degrade as real-world data drifts, then recover after retraining.

    > **Run `python scripts/simulate_timeline.py` then `python scripts/retrain_model.py`
    > to generate the data for this page.**
    """)

    # Check data availability
    if not os.path.exists(COMPARISON_PATH):
        st.warning(
            "⚠️  Timeline data not found. "
            "Run `python scripts/simulate_timeline.py` and `python scripts/retrain_model.py` first."
        )
        st.stop()

    comp_df    = pd.read_csv(COMPARISON_PATH)
    combined   = pd.read_csv(COMBINED_PATH) if os.path.exists(COMBINED_PATH) else None

    # ── Business Story ────────────────────────────────────────────────────────
    st.subheader("📖 The Business Story")
    story_data = {
        "Period": ["Jan 2023", "Jun 2023", "Dec 2023", "Mar 2024"],
        "Event":  [
            "Healthy company — model trained here",
            "Salary freeze + overwork begins",
            "Layoffs peak — extreme drift",
            "Recovery: raises, new hires — retrain here",
        ],
        "Attrition Rate": ["~18%", "~38%", "~53%", "~14%"],
        "Model Status": ["✅ Trained", "⚠️ Drifting", "❌ Failing", "✅ Retrained"],
    }
    st.table(pd.DataFrame(story_data))

    st.markdown("---")

    # ── Accuracy timeline chart ───────────────────────────────────────────────
    st.subheader("📉 Accuracy Over Time: Original vs Retrained Model")

    periods_short = ["Jan-23\n(Baseline)", "Jun-23\n(Stress)", "Dec-23\n(Crisis)", "Mar-24\n(Recovery)"]
    x = np.arange(len(periods_short))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, comp_df["orig_accuracy"],    "o-", color="#4C72B0", linewidth=2.5,
            markersize=9, label="Original Model (trained Jan-2023)")
    ax.plot(x, comp_df["retrain_accuracy"], "s--", color="#55A868", linewidth=2.5,
            markersize=9, label="Retrained Model (trained Mar-2024)")
    ax.axhline(0.75, color="red", linestyle=":", alpha=0.7, label="Acceptable threshold (0.75)")

    # Shade the crisis zone
    ax.axvspan(0.5, 2.5, alpha=0.08, color="red", label="Drift zone")

    # Annotate retrain point
    ax.annotate(
        "🔄 Retrain triggered\n(new data available)",
        xy=(3, comp_df["retrain_accuracy"].iloc[3]),
        xytext=(2.3, comp_df["retrain_accuracy"].iloc[3] - 0.12),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9, color="green",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(periods_short, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Model Accuracy Across Time Periods", fontsize=13)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── Side-by-side metric comparison ───────────────────────────────────────
    st.subheader("📊 Detailed Metric Comparison")
    display_comp = comp_df.copy()
    display_comp.columns = ["Period", "Orig Accuracy", "Orig F1", "Retrained Accuracy", "Retrained F1"]
    display_comp["Δ Accuracy"] = (display_comp["Retrained Accuracy"] - display_comp["Orig Accuracy"]).round(4)

    def color_delta(val):
        if val > 0:   return "background-color: #ccffcc"
        if val < 0:   return "background-color: #ffcccc"
        return ""

    st.dataframe(
        display_comp.style.applymap(color_delta, subset=["Δ Accuracy"]),
        use_container_width=True
    )

    st.markdown("---")

    # ── Attrition rate over time ──────────────────────────────────────────────
    st.subheader("📈 Attrition Rate Shift Across Periods")
    if combined is not None:
        agg = combined.groupby("period")["attrition"].mean().reset_index()
        # Sort by calendar order
        order = ["Jan-2023 (Baseline)", "Jun-2023 (Stress)", "Dec-2023 (Crisis)", "Mar-2024 (Recovery)"]
        agg["period"] = pd.Categorical(agg["period"], categories=order, ordered=True)
        agg = agg.sort_values("period")

        fig, ax = plt.subplots(figsize=(8, 4))
        colours = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]
        bars = ax.bar(agg["period"], agg["attrition"] * 100, color=colours, alpha=0.85, edgecolor="white")
        ax.axhline(30, color="red", linestyle="--", alpha=0.6, label="Warning threshold 30%")
        for bar, val in zip(bars, agg["attrition"] * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_title("True Attrition Rate by Time Period")
        ax.set_xticklabels(agg["period"], rotation=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Feature distribution shift ────────────────────────────────────────────
    st.subheader("🔬 How Salary & Hours Shifted Over Time")
    if combined is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colours_map = {
            "Jan-2023 (Baseline)":  "#4C72B0",
            "Jun-2023 (Stress)":    "#DD8452",
            "Dec-2023 (Crisis)":    "#C44E52",
            "Mar-2024 (Recovery)":  "#55A868",
        }
        for period, colour in colours_map.items():
            subset = combined[combined["period"] == period]
            axes[0].hist(subset["salary"],         bins=25, alpha=0.55, label=period, color=colour)
            axes[1].hist(subset["hours_per_week"],  bins=20, alpha=0.55, label=period, color=colour)

        axes[0].set_title("Salary Distribution Over Time")
        axes[0].set_xlabel("Salary ($)")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Hours/Week Distribution Over Time")
        axes[1].set_xlabel("Hours per Week")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "💡  **Key Takeaway**: The original model was trained on balanced, well-paying "
        "employees (Jan-2023). As salaries collapsed and hours ballooned in the crisis, "
        "the input data looked completely different from training data — the model had "
        "never seen such patterns and its accuracy dropped to ~60%. "
        "Retraining on the recovery data (which reflected the new normal) brought "
        "accuracy back above 85% on current data."
    )

