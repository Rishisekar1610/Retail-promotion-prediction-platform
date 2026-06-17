"""
============================================================
ROI & CAMPAIGN PERFORMANCE ANALYSIS
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
    logger.info("Loading A/B test and customer data from Supabase...")

    query = """
        SELECT
            a.customer_id,
            a.campaign_id,
            a."group",
            a.converted,
            a.conversion_probability,
            c.total_spend,
            c.predicted_clv,
            c.segment,
            c.clv_tier
        FROM ab_test_customer_assignments a
        LEFT JOIN customer_clv c
            ON a.customer_id = c.customer_id
    """

    df = pd.read_sql(query, engine)

    logger.info(f"Loaded {len(df):,} campaign customer records")

    return df


def simulate_campaign_revenue(df):
    logger.info("Simulating campaign revenue response...")

    np.random.seed(42)

    df = df.copy()

    df["baseline_revenue"] = df["total_spend"]

    df["campaign_revenue"] = df["baseline_revenue"]

    treatment_mask = df["group"] == "Treatment"
    control_mask = df["group"] == "Control"

    df.loc[control_mask, "campaign_revenue"] = (
        df.loc[control_mask, "baseline_revenue"]
        * np.random.uniform(
            0.95,
            1.05,
            control_mask.sum()
        )
    )

    df.loc[treatment_mask, "campaign_revenue"] = (
        df.loc[treatment_mask, "baseline_revenue"]
        * np.random.uniform(
            1.08,
            1.25,
            treatment_mask.sum()
        )
    )

    df["incremental_customer_revenue"] = (
        df["campaign_revenue"] - df["baseline_revenue"]
    )

    return df


def calculate_revenue_metrics(df):
    control = df[df["group"] == "Control"]
    treatment = df[df["group"] == "Treatment"]

    control_revenue = control["campaign_revenue"].sum()
    treatment_revenue = treatment["campaign_revenue"].sum()

    control_baseline_revenue = control["baseline_revenue"].sum()
    treatment_baseline_revenue = treatment["baseline_revenue"].sum()

    treatment_incremental_revenue = (
        treatment_revenue - treatment_baseline_revenue
    )

    return {
        "control_revenue": control_revenue,
        "treatment_revenue": treatment_revenue,
        "control_baseline_revenue": control_baseline_revenue,
        "treatment_baseline_revenue": treatment_baseline_revenue,
        "incremental_revenue": treatment_incremental_revenue,
        "control_customers": len(control),
        "treatment_customers": len(treatment),
        "control_conversions": int(control["converted"].sum()),
        "treatment_conversions": int(treatment["converted"].sum())
    }


def calculate_campaign_cost(treatment_customers):
    offer_cost_per_customer = 3.00

    campaign_cost = treatment_customers * offer_cost_per_customer

    return campaign_cost, offer_cost_per_customer


def calculate_roi_metrics(revenue_metrics, campaign_cost):
    incremental_revenue = revenue_metrics["incremental_revenue"]

    incremental_profit = incremental_revenue - campaign_cost

    roi = (incremental_profit / campaign_cost) * 100 if campaign_cost else 0

    roas = (
        revenue_metrics["treatment_revenue"] / campaign_cost
        if campaign_cost
        else 0
    )

    return incremental_profit, roi, roas


def calculate_cpa(campaign_cost, treatment_conversions):
    if treatment_conversions == 0:
        return 0

    return campaign_cost / treatment_conversions


def revenue_per_customer(revenue, customers):
    if customers == 0:
        return 0

    return revenue / customers


def create_visualization(control_revenue, treatment_revenue):
    fig = go.Figure()

    fig.add_bar(
        x=["Control"],
        y=[control_revenue],
        name="Control"
    )

    fig.add_bar(
        x=["Treatment"],
        y=[treatment_revenue],
        name="Treatment"
    )

    fig.update_layout(
        title="Campaign Revenue Comparison",
        yaxis_title="Revenue"
    )

    output_file = REPORT_DIR / "roi_analysis.html"

    fig.write_html(output_file)

    logger.info(f"Saved: {output_file}")


def segment_roi_report(df):
    report = (
        df.groupby(["segment", "group"])
        .agg(
            customers=("customer_id", "count"),
            avg_campaign_revenue=("campaign_revenue", "mean"),
            total_campaign_revenue=("campaign_revenue", "sum"),
            avg_incremental_revenue=("incremental_customer_revenue", "mean"),
            conversions=("converted", "sum")
        )
        .round(2)
        .reset_index()
    )

    output_file = REPORT_DIR / "roi_by_segment.csv"

    report.to_csv(output_file, index=False)

    report.to_sql(
        "roi_by_segment",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("roi_by_segment table created successfully.")

    return report


def save_customer_level_roi(df):
    output_df = df[[
        "customer_id",
        "campaign_id",
        "group",
        "segment",
        "clv_tier",
        "baseline_revenue",
        "campaign_revenue",
        "incremental_customer_revenue",
        "converted"
    ]]

    output_df.to_sql(
        "customer_roi_results",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("customer_roi_results table created successfully.")


def save_report(metrics):
    report = pd.DataFrame([metrics])

    output_file = REPORT_DIR / "roi_metrics.csv"

    report.to_csv(output_file, index=False)

    report.to_sql(
        "roi_metrics",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info(f"Saved: {output_file}")
    logger.info("roi_metrics table created successfully.")


def run_roi_analysis():
    logger.info("=" * 60)
    logger.info("ROI ANALYSIS PIPELINE")
    logger.info("=" * 60)

    df = load_data()

    df = simulate_campaign_revenue(df)

    revenue_metrics = calculate_revenue_metrics(df)

    campaign_cost, offer_cost = calculate_campaign_cost(
        revenue_metrics["treatment_customers"]
    )

    incremental_profit, roi, roas = calculate_roi_metrics(
        revenue_metrics,
        campaign_cost
    )

    cpa = calculate_cpa(
        campaign_cost,
        revenue_metrics["treatment_conversions"]
    )

    rpc_control = revenue_per_customer(
        revenue_metrics["control_revenue"],
        revenue_metrics["control_customers"]
    )

    rpc_treatment = revenue_per_customer(
        revenue_metrics["treatment_revenue"],
        revenue_metrics["treatment_customers"]
    )

    create_visualization(
        revenue_metrics["control_revenue"],
        revenue_metrics["treatment_revenue"]
    )

    metrics = {
        "campaign_id": "PROMO_001",
        "offer_cost_per_customer": round(offer_cost, 2),
        "control_customers": revenue_metrics["control_customers"],
        "treatment_customers": revenue_metrics["treatment_customers"],
        "control_conversions": revenue_metrics["control_conversions"],
        "treatment_conversions": revenue_metrics["treatment_conversions"],
        "control_revenue": round(revenue_metrics["control_revenue"], 2),
        "treatment_revenue": round(revenue_metrics["treatment_revenue"], 2),
        "control_baseline_revenue": round(
            revenue_metrics["control_baseline_revenue"],
            2
        ),
        "treatment_baseline_revenue": round(
            revenue_metrics["treatment_baseline_revenue"],
            2
        ),
        "incremental_revenue": round(
            revenue_metrics["incremental_revenue"],
            2
        ),
        "campaign_cost": round(campaign_cost, 2),
        "incremental_profit": round(incremental_profit, 2),
        "roi_percent": round(roi, 2),
        "roas": round(roas, 2),
        "cost_per_acquisition": round(cpa, 2),
        "control_revenue_per_customer": round(rpc_control, 2),
        "treatment_revenue_per_customer": round(rpc_treatment, 2)
    }

    save_customer_level_roi(df)

    segment_report = segment_roi_report(df)

    save_report(metrics)

    print("\nROI ANALYSIS RESULTS\n")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nROI BY SEGMENT\n")
    print(segment_report)

    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    run_roi_analysis()