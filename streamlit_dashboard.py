"""
============================================================
STREAMLIT BUSINESS DASHBOARD
Retail Promotion Analytics & Offer Redemption Prediction
============================================================
"""

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine
import plotly.express as px


st.set_page_config(
    page_title="Retail Promotion Analytics",
    page_icon="📊",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

if not DATABASE_URL:
    st.error("SUPABASE_DB_URL not found in .env file.")
    st.stop()

engine = create_engine(DATABASE_URL)

MODEL_DIR = BASE_DIR / "models"


@st.cache_data
def load_table(table_name):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "prediction_scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    best_model_name = joblib.load(MODEL_DIR / "best_model_name.pkl")

    return model, scaler, feature_names, best_model_name


df = load_table("customer_clv")
model, scaler, feature_names, best_model_name = load_artifacts()


st.sidebar.title("Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Customer Segmentation",
        "Customer Lifetime Value",
        "Offer Prediction",
        "A/B Testing",
        "ROI Analysis",
        "Model Monitoring"
    ]
)


if page == "Executive Overview":

    st.title("Executive Overview")

    total_customers = len(df)
    total_revenue = df["total_spend"].sum()
    avg_clv = df["predicted_clv"].mean()
    avg_frequency = df["purchase_frequency"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Revenue", f"${total_revenue:,.0f}")
    c3.metric("Avg Predicted CLV", f"${avg_clv:,.0f}")
    c4.metric("Avg Purchase Frequency", round(avg_frequency, 2))

    st.subheader("Revenue Distribution")

    fig = px.histogram(
        df,
        x="total_spend",
        nbins=50,
        title="Customer Revenue Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CLV Tier Distribution")

    tier_counts = df["clv_tier"].value_counts().reset_index()
    tier_counts.columns = ["clv_tier", "customers"]

    fig = px.bar(
        tier_counts,
        x="clv_tier",
        y="customers",
        title="Customers by CLV Tier"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    try:
        segments = load_table("customer_segments")
    except Exception:
        st.warning("customer_segments table not found.")
        st.stop()

    segment_counts = (
        segments["segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = ["segment", "customers"]

    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            segment_counts,
            names="segment",
            values="customers",
            title="Customer Segment Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(segment_counts, use_container_width=True)

    if "pca_1" in segments.columns and "pca_2" in segments.columns:

        fig = px.scatter(
            segments,
            x="pca_1",
            y="pca_2",
            color="segment",
            hover_data=["customer_id", "total_spend", "purchase_frequency"],
            title="PCA Customer Segmentation"
        )

        st.plotly_chart(fig, use_container_width=True)


elif page == "Customer Lifetime Value":

    st.title("Customer Lifetime Value")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(
            df,
            x="clv_tier",
            y="predicted_clv",
            title="Predicted CLV by Tier"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_summary = (
            df.groupby("clv_tier")
            .agg(
                customers=("customer_id", "count"),
                avg_clv=("predicted_clv", "mean"),
                avg_spend=("total_spend", "mean")
            )
            .round(2)
            .reset_index()
        )

        st.dataframe(tier_summary, use_container_width=True)

    st.subheader("Top Customers by Predicted CLV")

    top_customers = (
        df.sort_values("predicted_clv", ascending=False)
        .head(20)
    )

    st.dataframe(
        top_customers[
            [
                "customer_id",
                "segment",
                "clv_tier",
                "total_spend",
                "purchase_frequency",
                "predicted_clv"
            ]
        ],
        use_container_width=True
    )


elif page == "Offer Prediction":

    st.title("Offer Redemption Prediction")

    st.info(f"Current model: {best_model_name}")

    input_values = {}

    for feature in feature_names:

        default_value = float(df[feature].median())

        input_values[feature] = st.number_input(
            feature,
            value=default_value
        )

    if st.button("Predict Redemption Probability"):

        features = pd.DataFrame([input_values])

        features = features[feature_names]

        if best_model_name == "LogisticRegression":
            model_input = scaler.transform(features)
        else:
            model_input = features

        probability = model.predict_proba(model_input)[:, 1][0]

        prediction = int(probability >= 0.50)

        st.metric(
            "Redemption Probability",
            f"{probability:.2%}"
        )

        st.write(
            "Prediction:",
            "Likely to Redeem" if prediction == 1 else "Unlikely to Redeem"
        )


elif page == "A/B Testing":

    st.title("A/B Testing Results")

    try:
        ab_results = load_table("ab_test_results")
        assignments = load_table("ab_test_customer_assignments")
    except Exception:
        st.warning("Run ab_testing.py first.")
        st.stop()

    st.dataframe(ab_results, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Control Rate",
        f"{ab_results['control_rate'].iloc[0]:.2%}"
    )

    c2.metric(
        "Treatment Rate",
        f"{ab_results['treatment_rate'].iloc[0]:.2%}"
    )

    c3.metric(
        "Relative Lift",
        f"{ab_results['relative_lift_percent'].iloc[0]:.2f}%"
    )

    conversion_summary = (
        assignments.groupby("group")["converted"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        conversion_summary,
        x="group",
        y="converted",
        title="Control vs Treatment Conversion Rate"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "ROI Analysis":

    st.title("ROI Analysis")

    try:
        roi = load_table("roi_metrics")
        roi_segment = load_table("roi_by_segment")
    except Exception:
        st.warning("Run roi_analysis.py first.")
        st.stop()

    st.dataframe(roi, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ROI",
        f"{roi['roi_percent'].iloc[0]:.2f}%"
    )

    c2.metric(
        "ROAS",
        round(roi["roas"].iloc[0], 2)
    )

    c3.metric(
        "Incremental Profit",
        f"${roi['incremental_profit'].iloc[0]:,.0f}"
    )

    fig = px.bar(
        roi_segment,
        x="segment",
        y="total_campaign_revenue",
        color="group",
        barmode="group",
        title="Campaign Revenue by Segment"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Model Monitoring":

    st.title("Model Monitoring")

    try:
        drift = load_table("feature_drift_report")
        performance = load_table("model_performance_monitoring")
    except Exception:
        st.warning("Run monitoring.py first.")
        st.stop()

    st.subheader("Feature Drift Report")

    st.dataframe(drift, use_container_width=True)

    fig = px.bar(
        drift,
        x="feature",
        y="psi",
        color="status",
        title="Feature Drift Using PSI"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Performance Monitoring")

    st.dataframe(performance, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.info("Retail Promotion Analytics Platform")