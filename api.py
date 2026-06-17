"""
============================================================
FASTAPI MODEL DEPLOYMENT
Retail Promotion Analytics & Offer Redemption Prediction
============================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


app = FastAPI(
    title="Offer Redemption Prediction API",
    description="Predict promotion redemption probability using customer behavior",
    version="1.0.0"
)


# api.py is in project root
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "prediction_scaler.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
best_model_name = joblib.load(MODEL_DIR / "best_model_name.pkl")


class CustomerFeatures(BaseModel):
    total_spend: float
    purchase_frequency: float
    customer_lifetime_value: float
    recency: float
    retention_rate: float
    annual_purchase_frequency: float
    average_order_value: float
    customer_lifespan: float
    predicted_clv: float


@app.get("/")
def home():
    return {
        "message": "Offer Redemption Prediction API",
        "status": "running",
        "model": best_model_name,
        "features": feature_names
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(feature_names)
    }


def build_input(customer: CustomerFeatures):
    data = pd.DataFrame([customer.model_dump()])

    data = data[feature_names]

    if best_model_name == "LogisticRegression":
        data_input = scaler.transform(data)
    else:
        data_input = data

    return data_input


@app.post("/predict")
def predict(customer: CustomerFeatures):
    data_input = build_input(customer)

    probability = model.predict_proba(data_input)[:, 1][0]

    prediction = int(probability >= 0.50)

    return {
        "redemption_probability": round(float(probability), 4),
        "prediction": prediction,
        "interpretation": (
            "Likely to Redeem"
            if prediction == 1
            else "Unlikely to Redeem"
        ),
        "model": best_model_name
    }


@app.post("/batch_predict")
def batch_predict(customers: list[CustomerFeatures]):
    rows = [customer.model_dump() for customer in customers]

    df = pd.DataFrame(rows)

    df = df[feature_names]

    if best_model_name == "LogisticRegression":
        data_input = scaler.transform(df)
    else:
        data_input = df

    probabilities = model.predict_proba(data_input)[:, 1]

    predictions = (probabilities >= 0.50).astype(int)

    results = []

    for i in range(len(df)):
        results.append({
            "customer_index": i,
            "redemption_probability": round(float(probabilities[i]), 4),
            "prediction": int(predictions[i]),
            "interpretation": (
                "Likely to Redeem"
                if int(predictions[i]) == 1
                else "Unlikely to Redeem"
            )
        })

    return {
        "model": best_model_name,
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )