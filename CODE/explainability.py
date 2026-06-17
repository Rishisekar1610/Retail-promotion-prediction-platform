"""
============================================================
MODEL EXPLAINABILITY MODULE
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
import shap
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

engine = create_engine(DATABASE_URL)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def load_dataset():
    logger.info("Loading customer data from Supabase...")

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


def load_feature_names():
    return joblib.load(
        MODEL_DIR / "feature_names.pkl"
    )


def load_scaler():
    scaler_file = MODEL_DIR / "prediction_scaler.pkl"

    if scaler_file.exists():
        return joblib.load(scaler_file)

    return None


def create_feature_matrix(df, feature_names):
    X = df[feature_names].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return X


def prepare_shap_input(X, model_name, scaler):
    if model_name == "LogisticRegression" and scaler is not None:
        X_transformed = scaler.transform(X)

        X_shap = pd.DataFrame(
            X_transformed,
            columns=X.columns,
            index=X.index
        )

        return X_shap

    return X


def build_explainer(model, model_name, X):
    logger.info(f"Building SHAP explainer for {model_name}...")

    if model_name in ["RandomForest", "XGBoost"]:
        return shap.TreeExplainer(model)

    if model_name == "LogisticRegression":
        return shap.LinearExplainer(
            model,
            X,
            feature_perturbation="interventional"
        )

    return shap.Explainer(model, X)


def calculate_shap_values(explainer, X, model_name):
    logger.info("Calculating SHAP values...")

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    return shap_values


def create_summary_plot(shap_values, X):
    logger.info("Creating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    output_file = REPORT_DIR / "shap_summary.png"

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    logger.info(f"Saved: {output_file}")


def create_bar_plot(shap_values, X):
    logger.info("Creating SHAP feature importance bar chart...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    output_file = REPORT_DIR / "shap_feature_importance.png"

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    logger.info(f"Saved: {output_file}")


def create_feature_importance_report(shap_values, X):
    importance = np.abs(shap_values).mean(axis=0)

    report = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    })

    report = report.sort_values(
        by="importance",
        ascending=False
    )

    output_file = REPORT_DIR / "feature_importance.csv"

    report.to_csv(
        output_file,
        index=False
    )

    logger.info(f"Saved: {output_file}")

    return report


def create_dependence_plot(shap_values, X, feature_name):
    logger.info(f"Creating dependence plot for {feature_name}")

    shap.dependence_plot(
        feature_name,
        shap_values,
        X,
        show=False
    )

    output_file = REPORT_DIR / f"{feature_name}_dependence.png"

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300
    )

    plt.close()

    logger.info(f"Saved: {output_file}")


def generate_top_feature_plots(report, shap_values, X):
    top_features = report["feature"].head(3).tolist()

    for feature in top_features:
        create_dependence_plot(
            shap_values,
            X,
            feature
        )


def generate_business_insights(report):
    top_features = report.head(5)

    insights = []

    for _, row in top_features.iterrows():
        insights.append({
            "feature": row["feature"],
            "importance": round(row["importance"], 4),
            "business_interpretation": (
                f"{row['feature']} is one of the strongest drivers "
                "of offer redemption probability in the trained model."
            )
        })

    insights_df = pd.DataFrame(insights)

    output_file = REPORT_DIR / "business_insights.csv"

    insights_df.to_csv(
        output_file,
        index=False
    )

    logger.info(f"Saved: {output_file}")

    return insights_df


def save_explainability_to_supabase(report):
    report.to_sql(
        "feature_importance",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("feature_importance table created successfully in Supabase.")


def run_explainability():
    logger.info("=" * 60)
    logger.info("MODEL EXPLAINABILITY")
    logger.info("=" * 60)

    df = load_dataset()

    model = load_model()

    model_name = load_model_name()

    scaler = load_scaler()

    feature_names = load_feature_names()

    X = create_feature_matrix(
        df,
        feature_names
    )

    X_shap = prepare_shap_input(
        X,
        model_name,
        scaler
    )

    explainer = build_explainer(
        model,
        model_name,
        X_shap
    )

    shap_values = calculate_shap_values(
        explainer,
        X_shap,
        model_name
    )

    create_summary_plot(
        shap_values,
        X_shap
    )

    create_bar_plot(
        shap_values,
        X_shap
    )

    report = create_feature_importance_report(
        shap_values,
        X_shap
    )

    generate_top_feature_plots(
        report,
        shap_values,
        X_shap
    )

    insights = generate_business_insights(
        report
    )

    save_explainability_to_supabase(
        report
    )

    logger.info("\nTop Features:")
    print(report.head(10))

    logger.info("\nBusiness Insights:")
    print(insights)

    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    run_explainability()