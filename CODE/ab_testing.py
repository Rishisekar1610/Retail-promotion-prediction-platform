"""
============================================================
A/B TESTING & INCREMENTAL LIFT ANALYSIS
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

from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

import plotly.graph_objects as go


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

DATABASE_URL = "postgresql://postgres.hlxxjzstmgozujarbbxk:6vCCLI04X7k52GF0@aws-1-us-east-1.pooler.supabase.com:5432/postgres"

if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DB_URL not found in .env file.")

engine = create_engine(DATABASE_URL)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    logger.info("Loading customer data from Supabase...")

    query = """
        SELECT
            c.customer_id,
            c.segment,
            c.clv_tier,
            c.total_spend,
            c.purchase_frequency,
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


def create_experiment_groups(df):
    logger.info("Creating treatment and control groups...")

    np.random.seed(42)

    df = df.copy()

    df["campaign_id"] = "PROMO_001"

    df["group"] = np.random.choice(
        ["Control", "Treatment"],
        size=len(df),
        p=[0.50, 0.50]
    )

    return df


def simulate_campaign_results(df):
    logger.info("Simulating campaign conversion results...")

    np.random.seed(42)

    df = df.copy()

    base_rate = 0.12
    treatment_lift = 0.06

    clv_boost = (
        df["predicted_clv"] / df["predicted_clv"].max()
    ) * 0.05

    frequency_boost = (
        df["purchase_frequency"] / df["purchase_frequency"].max()
    ) * 0.04

    df["conversion_probability"] = (
        base_rate
        + clv_boost
        + frequency_boost
    )

    df.loc[
        df["group"] == "Treatment",
        "conversion_probability"
    ] += treatment_lift

    df["conversion_probability"] = df["conversion_probability"].clip(
        lower=0.02,
        upper=0.80
    )

    df["converted"] = (
        np.random.rand(len(df)) < df["conversion_probability"]
    ).astype(int)

    return df


def calculate_conversion_rates(df):
    report = (
        df.groupby("group")["converted"]
        .agg(
            conversions="sum",
            customers="count",
            conversion_rate="mean"
        )
    )

    return report


def calculate_lift(report):
    control_rate = report.loc["Control", "conversion_rate"]
    treatment_rate = report.loc["Treatment", "conversion_rate"]

    absolute_lift = treatment_rate - control_rate
    relative_lift = (absolute_lift / control_rate) * 100

    return control_rate, treatment_rate, absolute_lift, relative_lift


def perform_z_test(df):
    treatment = df[df["group"] == "Treatment"]
    control = df[df["group"] == "Control"]

    successes = np.array([
        treatment["converted"].sum(),
        control["converted"].sum()
    ])

    observations = np.array([
        len(treatment),
        len(control)
    ])

    z_stat, p_value = proportions_ztest(
        successes,
        observations
    )

    return z_stat, p_value


def perform_chi_square(df):
    contingency = pd.crosstab(
        df["group"],
        df["converted"]
    )

    chi2, p, _, _ = chi2_contingency(contingency)

    return chi2, p


def confidence_interval(
    control_rate,
    treatment_rate,
    control_n,
    treatment_n
):
    diff = treatment_rate - control_rate

    se = np.sqrt(
        (treatment_rate * (1 - treatment_rate)) / treatment_n
        +
        (control_rate * (1 - control_rate)) / control_n
    )

    margin = 1.96 * se

    lower = diff - margin
    upper = diff + margin

    return lower, upper


def create_visualization(control_rate, treatment_rate):
    fig = go.Figure()

    fig.add_bar(
        x=["Control"],
        y=[control_rate],
        name="Control"
    )

    fig.add_bar(
        x=["Treatment"],
        y=[treatment_rate],
        name="Treatment"
    )

    fig.update_layout(
        title="Campaign Conversion Rates",
        yaxis_title="Conversion Rate"
    )

    output_file = REPORT_DIR / "ab_test_results.html"

    fig.write_html(output_file)

    logger.info(f"Saved: {output_file}")


def save_customer_level_experiment(df):
    experiment_df = df[[
        "customer_id",
        "campaign_id",
        "group",
        "conversion_probability",
        "converted"
    ]]

    experiment_df.to_sql(
        "ab_test_customer_assignments",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("ab_test_customer_assignments table created successfully.")


def save_report(results):
    report = pd.DataFrame([results])

    output_file = REPORT_DIR / "ab_test_metrics.csv"

    report.to_csv(output_file, index=False)

    report.to_sql(
        "ab_test_results",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info(f"Saved: {output_file}")
    logger.info("ab_test_results table created successfully.")


def run_ab_test():
    logger.info("=" * 60)
    logger.info("A/B TESTING PIPELINE")
    logger.info("=" * 60)

    df = load_data()

    df = create_experiment_groups(df)

    df = simulate_campaign_results(df)

    report = calculate_conversion_rates(df)

    (
        control_rate,
        treatment_rate,
        absolute_lift,
        relative_lift
    ) = calculate_lift(report)

    z_stat, z_p = perform_z_test(df)

    chi2, chi_p = perform_chi_square(df)

    control_n = len(df[df["group"] == "Control"])
    treatment_n = len(df[df["group"] == "Treatment"])

    lower_ci, upper_ci = confidence_interval(
        control_rate,
        treatment_rate,
        control_n,
        treatment_n
    )

    create_visualization(
        control_rate,
        treatment_rate
    )

    results = {
        "campaign_id": "PROMO_001",
        "control_customers": control_n,
        "treatment_customers": treatment_n,
        "control_conversions": int(
            df[df["group"] == "Control"]["converted"].sum()
        ),
        "treatment_conversions": int(
            df[df["group"] == "Treatment"]["converted"].sum()
        ),
        "control_rate": round(control_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "absolute_lift": round(absolute_lift, 4),
        "relative_lift_percent": round(relative_lift, 2),
        "z_statistic": round(z_stat, 4),
        "z_test_p_value": round(z_p, 6),
        "chi_square": round(chi2, 4),
        "chi_square_p": round(chi_p, 6),
        "lower_ci": round(lower_ci, 4),
        "upper_ci": round(upper_ci, 4),
        "statistically_significant": bool(z_p < 0.05)
    }

    save_customer_level_experiment(df)

    save_report(results)

    print("\nA/B TEST RESULTS\n")

    for k, v in results.items():
        print(f"{k}: {v}")

    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_ab_test()