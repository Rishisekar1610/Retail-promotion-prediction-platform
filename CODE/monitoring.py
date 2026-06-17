"""
============================================================
MODEL MONITORING & DRIFT DETECTION
Retail Promotion Analytics & Offer Redemption Prediction
============================================================
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

import joblib

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

import plotly.graph_objects as go


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DB_URL not found in .env file.")

engine = create_engine(DATABASE_URL)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    logger.info("Loading monitoring data from Supabase...")

    query = """
        SELECT
            c.customer_id,
            c.total_spend,
            c.purchase_frequency,
            c.customer_lifetime_value,
            c.recency,
            c.retention_rate,
            c.annual_purchase_frequency,
            c.average_order_value,
            c.customer_lifespan,
            c.predicted_clv,
            f.redeemed_offer
        FROM customer_clv c
        LEFT JOIN customer_features f
            ON c.customer_id = f.customer_id
        WHERE f.redeemed_offer IS NOT NULL
    """

    df = pd.read_sql(query, engine)

    logger.info(f"Loaded {len(df):,} customers")

    return df


def load_model():
    logger.info("Loading trained model...")

    return joblib.load(
        MODEL_DIR / "best_model.pkl"
    )


def load_model_name():
    model_name_file = MODEL_DIR / "best_model_name.pkl"

    if model_name_file.exists():
        return joblib.load(model_name_file)

    return "Unknown"


def load_scaler():
    scaler_file = MODEL_DIR / "prediction_scaler.pkl"

    if scaler_file.exists():
        return joblib.load(scaler_file)

    return None


def load_feature_names():
    return joblib.load(
        MODEL_DIR / "feature_names.pkl"
    )


def create_production_dataset(df):
    logger.info("Creating simulated production dataset...")

    prod_df = df.copy()

    np.random.seed(42)

    prod_df["total_spend"] *= np.random.uniform(
        1.05,
        1.25,
        len(prod_df)
    )

    prod_df["purchase_frequency"] *= np.random.uniform(
        0.90,
        1.20,
        len(prod_df)
    )

    prod_df["average_order_value"] = (
        prod_df["total_spend"]
        / prod_df["purchase_frequency"].replace(0, np.nan)
    ).fillna(0)

    prod_df["predicted_clv"] *= np.random.uniform(
        1.03,
        1.18,
        len(prod_df)
    )

    return prod_df


def clean_feature_matrix(df, feature_names):
    X = df[feature_names].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return X


def transform_if_needed(X, model_name, scaler):
    if model_name == "LogisticRegression" and scaler is not None:
        return scaler.transform(X)

    return X


def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0

    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, buckets + 1)
    )

    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(
        expected_pct == 0,
        0.0001,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        0.0001,
        actual_pct
    )

    psi = np.sum(
        (expected_pct - actual_pct)
        * np.log(expected_pct / actual_pct)
    )

    return psi


def calculate_feature_drift(reference_df, production_df, feature_names):
    drift_results = []

    for feature in feature_names:
        psi = calculate_psi(
            reference_df[feature],
            production_df[feature]
        )

        if psi < 0.10:
            status = "Stable"
        elif psi < 0.20:
            status = "Moderate Drift"
        else:
            status = "Significant Drift"

        drift_results.append({
            "feature": feature,
            "psi": round(float(psi), 4),
            "status": status
        })

    drift_df = pd.DataFrame(drift_results)

    return drift_df


def evaluate_model(model, df, feature_names, model_name, scaler):
    X = clean_feature_matrix(
        df,
        feature_names
    )

    X_input = transform_if_needed(
        X,
        model_name,
        scaler
    )

    y = df["redeemed_offer"].astype(int)

    probabilities = model.predict_proba(X_input)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "avg_prediction_probability": float(np.mean(probabilities))
    }

    return metrics


def compare_performance(baseline, production):
    comparison = []

    for metric in baseline:
        degradation = baseline[metric] - production[metric]

        comparison.append({
            "metric": metric,
            "baseline": round(float(baseline[metric]), 4),
            "production": round(float(production[metric]), 4),
            "degradation": round(float(degradation), 4)
        })

    return pd.DataFrame(comparison)


def create_monitoring_dashboard(drift_df):
    fig = go.Figure()

    fig.add_bar(
        x=drift_df["feature"],
        y=drift_df["psi"]
    )

    fig.update_layout(
        title="Feature Drift Monitoring Using PSI",
        yaxis_title="Population Stability Index"
    )

    output_file = REPORT_DIR / "monitoring_dashboard.html"

    fig.write_html(output_file)

    logger.info(f"Saved: {output_file}")


def generate_alerts(drift_df):
    alerts = drift_df[
        drift_df["psi"] > 0.20
    ].copy()

    if len(alerts) > 0:
        logger.warning("SIGNIFICANT DRIFT DETECTED")
        print("\nDRIFT ALERTS\n")
        print(alerts)

    return alerts


def save_reports(drift_df, performance_df, alerts):
    drift_output = REPORT_DIR / "feature_drift_report.csv"
    perf_output = REPORT_DIR / "performance_monitoring.csv"
    alert_output = REPORT_DIR / "drift_alerts.csv"

    drift_df.to_csv(drift_output, index=False)
    performance_df.to_csv(perf_output, index=False)
    alerts.to_csv(alert_output, index=False)

    drift_df.to_sql(
        "feature_drift_report",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    performance_df.to_sql(
        "model_performance_monitoring",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    alerts.to_sql(
        "drift_alerts",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info(f"Saved: {drift_output}")
    logger.info(f"Saved: {perf_output}")
    logger.info(f"Saved: {alert_output}")
    logger.info("Monitoring tables created successfully in Supabase.")


def run_monitoring():
    logger.info("=" * 60)
    logger.info("MODEL MONITORING")
    logger.info("=" * 60)

    reference_df = load_data()

    production_df = create_production_dataset(
        reference_df
    )

    model = load_model()

    model_name = load_model_name()

    scaler = load_scaler()

    feature_names = load_feature_names()

    drift_df = calculate_feature_drift(
        reference_df,
        production_df,
        feature_names
    )

    baseline_metrics = evaluate_model(
        model,
        reference_df,
        feature_names,
        model_name,
        scaler
    )

    production_metrics = evaluate_model(
        model,
        production_df,
        feature_names,
        model_name,
        scaler
    )

    performance_df = compare_performance(
        baseline_metrics,
        production_metrics
    )

    create_monitoring_dashboard(
        drift_df
    )

    alerts = generate_alerts(
        drift_df
    )

    save_reports(
        drift_df,
        performance_df,
        alerts
    )

    print("\nFEATURE DRIFT\n")
    print(drift_df)

    print("\nMODEL PERFORMANCE\n")
    print(performance_df)

    logger.info("=" * 60)

    return drift_df, performance_df


if __name__ == "__main__":
    run_monitoring()