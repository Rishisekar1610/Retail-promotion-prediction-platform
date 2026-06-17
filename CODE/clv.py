"""
============================================================
CUSTOMER LIFETIME VALUE MODULE
Retail Promotion Analytics & Offer Redemption Prediction
============================================================
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")
engine = create_engine(DATABASE_URL)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    logger.info("Loading customer_features + customer_segments from Supabase...")

    query = """
        SELECT
            f.customer_id,
            f.total_spend,
            f.avg_basket_size,
            f.purchase_frequency,
            f.recency,
            f.customer_lifetime_value,
            f.rfm_score,
            s.segment_id,
            s.segment
        FROM customer_features f
        LEFT JOIN customer_segments s
            ON f.customer_id = s.customer_id
    """

    df = pd.read_sql(query, engine)

    logger.info(f"Loaded {len(df):,} customers")

    return df


def calculate_retention_rate(df):
    max_recency = df["recency"].max()

    df["retention_rate"] = 1 - (df["recency"] / max_recency)

    df["retention_rate"] = df["retention_rate"].clip(
        lower=0.05,
        upper=0.95
    )

    return df


def calculate_purchase_frequency(df):
    df["annual_purchase_frequency"] = df["purchase_frequency"] * 12
    return df


def calculate_aov(df):
    df["average_order_value"] = (
        df["total_spend"] / df["purchase_frequency"]
    )

    df["average_order_value"] = df["average_order_value"].replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    return df


def estimate_customer_lifespan(df):
    df["customer_lifespan"] = 1 / (1 - df["retention_rate"])
    return df


def calculate_clv(df):
    df["predicted_clv"] = (
        df["average_order_value"]
        * df["annual_purchase_frequency"]
        * df["customer_lifespan"]
    )

    return df


def create_clv_tiers(df):
    q1 = df["predicted_clv"].quantile(0.25)
    q2 = df["predicted_clv"].quantile(0.50)
    q3 = df["predicted_clv"].quantile(0.75)

    conditions = [
        df["predicted_clv"] >= q3,
        (df["predicted_clv"] >= q2) & (df["predicted_clv"] < q3),
        (df["predicted_clv"] >= q1) & (df["predicted_clv"] < q2)
    ]

    labels = ["Premium", "High", "Medium"]

    df["clv_tier"] = np.select(
        conditions,
        labels,
        default="Low"
    )

    return df


def segment_clv_report(df):
    report = (
        df.groupby("segment")
        .agg({
            "predicted_clv": "mean",
            "customer_id": "count"
        })
        .round(2)
        .reset_index()
    )

    report.columns = [
        "segment",
        "avg_clv",
        "customers"
    ]

    output_file = REPORT_DIR / "clv_by_segment.csv"

    report.to_csv(output_file, index=False)

    logger.info(f"Saved: {output_file}")

    return report


def clv_tier_report(df):
    report = (
        df.groupby("clv_tier")
        .agg({
            "predicted_clv": "mean",
            "customer_id": "count"
        })
        .round(2)
        .reset_index()
    )

    report.columns = [
        "clv_tier",
        "avg_clv",
        "customers"
    ]

    output_file = REPORT_DIR / "clv_tiers.csv"

    report.to_csv(output_file, index=False)

    logger.info(f"Saved: {output_file}")

    return report


def save_data(df):
    clv_df = df[[
        "customer_id",
        "segment",
        "segment_id",
        "total_spend",
        "purchase_frequency",
        "customer_lifetime_value",
        "recency",
        "retention_rate",
        "annual_purchase_frequency",
        "average_order_value",
        "customer_lifespan",
        "predicted_clv",
        "clv_tier"
    ]]

    clv_df.to_sql(
        "customer_clv",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("customer_clv table created successfully.")


def run_clv_pipeline():
    logger.info("=" * 60)
    logger.info("CUSTOMER LIFETIME VALUE PIPELINE")
    logger.info("=" * 60)

    df = load_data()

    df = calculate_retention_rate(df)
    df = calculate_purchase_frequency(df)
    df = calculate_aov(df)
    df = estimate_customer_lifespan(df)
    df = calculate_clv(df)
    df = create_clv_tiers(df)

    segment_report = segment_clv_report(df)
    tier_report = clv_tier_report(df)

    save_data(df)

    logger.info("\nCLV BY SEGMENT\n")
    print(segment_report)

    logger.info("\nCLV BY TIER\n")
    print(tier_report)

    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    clv_df = run_clv_pipeline()

    print(
        clv_df[
            [
                "customer_id",
                "segment",
                "predicted_clv",
                "clv_tier"
            ]
        ].head()
    )