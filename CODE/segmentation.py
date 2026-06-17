import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import plotly.express as px
import joblib


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

engine = create_engine(DATABASE_URL)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    logger.info("Loading customer_features from Supabase...")

    df = pd.read_sql(
        "SELECT * FROM customer_features",
        engine
    )

    logger.info(f"Loaded {len(df):,} customers")

    return df


def select_features(df):
    features = [
        "total_spend",
        "avg_basket_size",
        "purchase_frequency",
        "unique_products",
        "customer_lifetime_value",
        "rfm_score"
    ]

    X = df[features].fillna(0)

    return X, features


def scale_features(X):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        MODEL_DIR / "segment_scaler.pkl"
    )

    logger.info("Scaler saved.")

    return X_scaled


def train_gmm(X_scaled):
    logger.info("Training Gaussian Mixture Model...")

    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42
    )

    labels = gmm.fit_predict(X_scaled)

    joblib.dump(
        gmm,
        MODEL_DIR / "gmm_segmentation.pkl"
    )

    logger.info("GMM model saved.")

    return gmm, labels


def map_segments(df):
    segment_stats = (
        df.groupby("segment_id")["customer_lifetime_value"]
        .mean()
        .sort_values(ascending=False)
    )

    ranking = segment_stats.index.tolist()

    mapping = {
        ranking[0]: "High Value",
        ranking[1]: "Loyal",
        ranking[2]: "Occasional",
        ranking[3]: "At Risk"
    }

    df["segment"] = df["segment_id"].map(mapping)

    return df


def create_visualization(df, X_scaled):
    logger.info("Creating PCA visualization...")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    coords = pca.fit_transform(X_scaled)

    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]

    viz_df = df[[
        "customer_id",
        "pca_1",
        "pca_2",
        "segment",
        "total_spend",
        "purchase_frequency",
        "customer_lifetime_value"
    ]]

    fig = px.scatter(
        viz_df,
        x="pca_1",
        y="pca_2",
        color="segment",
        hover_data=[
            "customer_id",
            "total_spend",
            "purchase_frequency",
            "customer_lifetime_value"
        ],
        title="Customer Segmentation Using Gaussian Mixture Model"
    )

    output_file = REPORT_DIR / "customer_segments.html"

    fig.write_html(output_file)

    logger.info(f"Visualization saved to {output_file}")

    return df


def generate_segment_report(df):
    report = (
        df.groupby("segment")
        .agg({
            "customer_id": "count",
            "total_spend": "mean",
            "purchase_frequency": "mean",
            "customer_lifetime_value": "mean",
            "rfm_score": "mean"
        })
        .round(2)
        .reset_index()
    )

    report.columns = [
        "segment",
        "customers",
        "avg_spend",
        "avg_frequency",
        "avg_clv",
        "avg_rfm_score"
    ]

    output_file = REPORT_DIR / "segment_summary.csv"

    report.to_csv(output_file, index=False)

    logger.info(f"Report saved to {output_file}")

    return report


def save_segmented_data(df):
    segment_df = df[[
        "customer_id",
        "segment_id",
        "segment",
        "pca_1",
        "pca_2",
        "total_spend",
        "purchase_frequency",
        "customer_lifetime_value",
        "rfm_score"
    ]]

    segment_df.to_sql(
        "customer_segments",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info("customer_segments table created successfully.")


def run_segmentation():
    logger.info("=" * 60)
    logger.info("CUSTOMER SEGMENTATION")
    logger.info("=" * 60)

    df = load_data()

    X, feature_names = select_features(df)

    X_scaled = scale_features(X)

    gmm, labels = train_gmm(X_scaled)

    df["segment_id"] = labels

    df = map_segments(df)

    df = create_visualization(df, X_scaled)

    report = generate_segment_report(df)

    save_segmented_data(df)

    logger.info("\nSegment Summary:\n")
    print(report)

    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    segmented_df = run_segmentation()

    print(
        segmented_df[
            [
                "customer_id",
                "segment_id",
                "segment"
            ]
        ].head()
    )