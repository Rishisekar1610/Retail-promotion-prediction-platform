"""
============================================================
OFFER REDEMPTION MODEL TRAINING
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
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


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


MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MLFLOW_DB = BASE_DIR / "mlflow.db"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
MLFLOW_DB = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
)


FEATURES = [
    "total_spend",
    "purchase_frequency",
    "customer_lifetime_value",
    "recency",
    "retention_rate",
    "annual_purchase_frequency",
    "average_order_value",
    "customer_lifespan",
    "predicted_clv"
]

TARGET = "redeemed_offer"


def load_data():
    logger.info("Loading model training data from Supabase...")

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


def prepare_data(df):
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X,
        y,
        df["customer_id"],
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(
        scaler,
        MODEL_DIR / "prediction_scaler.pkl"
    )

    joblib.dump(
        FEATURES,
        MODEL_DIR / "feature_names.pkl"
    )

    logger.info("Scaler and feature names saved.")

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        test_ids
    )


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0)
    }

    return metrics, probabilities, predictions


def train_models(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):
    mlflow.set_experiment(
        "retail_offer_redemption_prediction"
    )

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ),
            "scaled": True
        },

        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=42,
                class_weight="balanced"
            ),
            "scaled": False
        },

        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                eval_metric="logloss",
                random_state=42
            ),
            "scaled": False
        }
    }

    best_model = None
    best_name = None
    best_auc = -1
    best_predictions = None
    best_probabilities = None

    results = []

    for name, config in models.items():

        logger.info(f"Training {name}...")

        model = config["model"]

        if config["scaled"]:
            model.fit(X_train_scaled, y_train)

            metrics, probabilities, predictions = evaluate_model(
                model,
                X_test_scaled,
                y_test
            )

        else:
            model.fit(X_train, y_train)

            metrics, probabilities, predictions = evaluate_model(
                model,
                X_test,
                y_test
            )

        with mlflow.start_run(run_name=name):

            mlflow.log_param("model_name", name)
            mlflow.log_param("num_features", len(FEATURES))

            for feature in FEATURES:
                mlflow.log_param(f"feature_{feature}", feature)

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            if name == "XGBoost":
                mlflow.xgboost.log_model(
                model,
                name=name
                )
            else:
                mlflow.sklearn.log_model(
                model,
                name=name
                )

        result = {
            "model": name,
            "roc_auc": round(metrics["roc_auc"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4)
        }

        results.append(result)

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)
        print(result)
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = model
            best_name = name
            best_predictions = predictions
            best_probabilities = probabilities

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "model_results.csv",
        index=False
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    joblib.dump(
        best_name,
        MODEL_DIR / "best_model_name.pkl"
    )

    logger.info(
        f"Best model saved: {best_name} | ROC-AUC: {best_auc:.4f}"
    )

    return (
        best_model,
        best_name,
        best_probabilities,
        best_predictions,
        results_df
    )


def save_predictions(
    test_ids,
    probabilities,
    predictions,
    best_name
):
    prediction_df = pd.DataFrame({
        "customer_id": test_ids.values,
        "redemption_probability": probabilities,
        "prediction": predictions,
        "model_name": best_name,
        "prediction_timestamp": pd.Timestamp.now()
    })

    prediction_df.to_sql(
        "model_predictions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50000,
        method="multi"
    )

    logger.info(
        "model_predictions table created successfully."
    )


def run_training_pipeline():
    logger.info("=" * 60)
    logger.info("OFFER REDEMPTION MODEL TRAINING")
    logger.info("=" * 60)

    df = load_data()

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        test_ids
    ) = prepare_data(df)

    (
        best_model,
        best_name,
        best_probabilities,
        best_predictions,
        results_df
    ) = train_models(
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    save_predictions(
        test_ids,
        best_probabilities,
        best_predictions,
        best_name
    )

    print("\nMODEL RESULTS")
    print(results_df)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()