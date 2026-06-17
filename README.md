<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />
<img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />

<br /><br />

# 🛍️ Retail Promotion Analytics Platform

### End-to-end ML system for offer redemption prediction, customer segmentation, A/B testing, and campaign ROI — built for production.

<br />

*Which customers redeem offers? Which campaigns generate ROI? How do you know when your model is drifting?*
*This platform answers all three — at scale.*

<br />

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📌 What This Solves

Retailers spend heavily on promotions but lack the tools to answer the questions that matter most:

| Question | This Platform's Answer |
|----------|------------------------|
| Which customers will redeem an offer? | XGBoost redemption model (AUC > 0.78) |
| Which segments drive the most value? | GMM clustering + CLV estimation |
| Did our campaign actually work? | A/B testing with z-test, chi-square, lift analysis |
| What's the real ROI of a promotion? | Full cost-revenue attribution formula per segment |
| Is our model still accurate in production? | PSI drift detection + AUC monitoring with alerts |

---

## 🏗️ System Architecture

```
Online Retail II Dataset (1M+ transactions)
             │
             ▼
   ┌─────────────────────┐
   │  PostgreSQL/Supabase │  ← Raw data layer
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │  PySpark Pipeline   │  ← Distributed feature engineering
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │  GMM Segmentation   │  ← High Value / Loyal / Occasional / At-Risk
   │  + CLV Estimation   │
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────────────────────────┐
   │  Offer Redemption Prediction            │
   │  LogReg · Random Forest · XGBoost       │
   │  SHAP Explainability · MLflow Tracking  │
   └─────────┬───────────────────────────────┘
             │
             ▼
   ┌──────────────────────────────────────┐
   │  A/B Testing  ·  ROI Analysis        │
   │  Drift Detection  ·  Alerting        │
   └─────────┬────────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 FastAPI        Streamlit
 REST API       Dashboard
```

---

## 📂 Project Structure

```
retail-promo-platform/
│
├── 📄 api.py                          # FastAPI prediction service
├── 📄 streamlit_dashboard.py          # Interactive business dashboard
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📄 requirements.txt
│
├── CODE/
│   ├── load_data_to_supabase.py       # Raw ingestion → PostgreSQL
│   ├── data_cleaning_supabase.py      # Documented cleaning pipeline
│   ├── feature_engineering.py         # Customer-level feature creation
│   ├── segmentation.py                # GMM clustering + PCA
│   ├── clv.py                         # Customer Lifetime Value
│   ├── training.py                    # Model training + MLflow
│   ├── explainability.py              # SHAP global + local
│   ├── ab_testing.py                  # Experiment design + stats
│   ├── roi_analysis.py                # Cost-revenue attribution
│   └── monitoring.py                  # PSI + AUC drift detection
│
├── models/
│   ├── best_model.pkl
│   ├── gmm_segmentation.pkl
│   ├── prediction_scaler.pkl
│   └── feature_names.pkl
│
├── reports/
│   ├── feature_importance.csv
│   ├── ab_test_metrics.csv
│   ├── roi_metrics.csv
│   ├── feature_drift_report.csv
│   └── monitoring_dashboard.html
│
└── notebooks/
    └── pyspark_feature_engineering.ipynb
```

---

## ⚙️ Tech Stack

<table>
<tr>
<td><b>Layer</b></td>
<td><b>Technologies</b></td>
</tr>
<tr>
<td>Language</td>
<td>Python 3.10+, SQL</td>
</tr>
<tr>
<td>Data Engineering</td>
<td>Pandas, NumPy, PySpark, SQLAlchemy, PostgreSQL, Supabase</td>
</tr>
<tr>
<td>Machine Learning</td>
<td>Scikit-learn, XGBoost, Gaussian Mixture Models, SHAP</td>
</tr>
<tr>
<td>MLOps</td>
<td>MLflow (experiment tracking + model registry), FastAPI, Docker</td>
</tr>
<tr>
<td>Visualization</td>
<td>Plotly, Streamlit</td>
</tr>
<tr>
<td>Statistics</td>
<td>SciPy (t-test, chi-square, z-test), confidence intervals</td>
</tr>
</table>

---

## 🔹 Pipeline Walkthrough

### 1 · Data Engineering

- Loaded 1M+ retail transactions into PostgreSQL (Supabase) with automated schema creation
- Documented every cleaning decision in a structured log: cancelled orders, invalid quantities, missing customer IDs, zero-price records
- PySpark distributed pipeline for feature engineering at scale

**Customer-level features engineered:**

```
Total Spend · Purchase Frequency · Avg Basket Size
Unique Products · Recency · RFM Score
Customer Lifetime Value · Redemption Probability
```

---

### 2 · Customer Segmentation

Implemented **Gaussian Mixture Model (GMM)** clustering — chosen over K-Means for its ability to model non-spherical cluster shapes and produce soft membership probabilities.

| Segment | Profile |
|---------|---------|
| 🟢 High Value | High frequency, high spend, recent purchasers |
| 🔵 Loyal | Consistent buyers, moderate spend |
| 🟡 Occasional | Infrequent, price-sensitive |
| 🔴 At-Risk | Lapsed customers, low engagement |

PCA visualizations + segment-level analytics generated for each cohort.

---

### 3 · Customer Lifetime Value (CLV)

```
CLV = Average Order Value × Purchase Frequency × Customer Lifespan
```

CLV tiers assigned:

```
Premium  ·  High  ·  Medium  ·  Low
```

---

### 4 · Offer Redemption Prediction

**Problem:** Binary classification — will a customer redeem a promotional offer?

Three models trained and compared:

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, interpretable coefficients |
| Random Forest | Handles nonlinear interactions |
| XGBoost | Best AUC; production model |

**Evaluation:** ROC-AUC · Accuracy · Precision · Recall · F1

All experiments tracked via **MLflow** (metrics, parameters, model artifacts, registry).

---

### 5 · Explainable AI (SHAP)

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)
```

Outputs:
- Global feature importance ranking
- SHAP summary plots (beeswarm + bar)
- Feature dependence plots
- Business-readable insight reports

---

### 6 · A/B Testing

Randomized control/treatment experiment design with full statistical testing:

```
Two-Proportion Z-Test  ·  Chi-Square Test  ·  Confidence Intervals
```

Metrics reported:

| Metric | Description |
|--------|-------------|
| Conversion Rate | Per-group purchase rate |
| Absolute Lift | Treatment rate − control rate |
| Relative Lift | % improvement over control |
| Statistical Significance | p-value vs α = 0.05 |

---

### 7 · Campaign ROI Analysis

Full cost-revenue attribution per campaign and customer segment:

```
Net Profit  = (Redeemed × AOV × Margin) − (Discount Cost + Send Cost)
ROI         = Net Profit / Total Promo Cost × 100
ROAS        = Gross Revenue / Total Promo Cost
Break-even  = Cost Per Send / (AOV × Margin − Discount)
```

Reports generated:
- `roi_metrics.csv` — per-campaign ROI breakdown
- Segment-level ROI analytics (High-Value vs Occasional vs At-Risk)
- Revenue attribution visualizations

---

### 8 · Model Monitoring & Drift Detection

**Population Stability Index (PSI)** computed per feature between training and production distributions:

| PSI Range | Status | Action |
|-----------|--------|--------|
| < 0.10 | ✅ Stable | Continue monitoring |
| 0.10 – 0.20 | ⚠️ Minor drift | Increase check frequency |
| > 0.20 | 🚨 Major drift | Retrain immediately |

Performance tracked continuously:

```
ROC-AUC  ·  Precision  ·  Recall  ·  F1  ·  Accuracy
```

Automated alerts fire when AUC drops below threshold or PSI exceeds critical level.

---

## 🌐 API Reference

FastAPI service with Swagger docs at `http://localhost:8000/docs`

```bash
# Health check
GET  /health

# Single customer prediction
POST /predict

# Batch scoring
POST /batch_predict
```

**Example request:**

```json
{
  "total_spend": 450.0,
  "avg_basket_size": 30.0,
  "purchase_frequency": 15,
  "unique_products": 20,
  "days_since_purchase": 10,
  "rfm_score": 9,
  "segment_code": 2
}
```

**Example response:**

```json
{
  "redemption_probability": 0.7341,
  "will_redeem": true,
  "segment": "High-Value",
  "model_used": "XGBClassifier"
}
```

---

## 📊 Streamlit Dashboard

Seven interactive views in a single dashboard:

```
Executive Overview     → KPIs: customers, redemption rate, revenue, ROI, AUC
Customer Segmentation  → RFM scatter, segment distribution, cohort summary
Customer Lifetime Value→ CLV tiers, ranked customer table
Offer Prediction       → Live API call, gauge chart, batch preview
A/B Testing            → Lift analysis, power curve, full stats report
ROI Analysis           → Sensitivity curves, break-even, segment ROI
Model Monitoring       → AUC over time, PSI per feature, drift alerts
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services started:
#   FastAPI   → http://localhost:8000
#   Streamlit → http://localhost:8501
#   MLflow    → http://localhost:5000
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/rishignanasekar/retail-promo-platform.git
cd retail-promo-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python retail_promo_platform.py

# 4. Start FastAPI
uvicorn api:app --reload

# 5. Start Streamlit (new terminal)
streamlit run streamlit_dashboard.py
```

> **Dataset:** Download the [Online Retail II dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) and place `online_retail_II.xlsx` in the project root.

---

## 👨‍💻 Author

**Rishi Gnanasekar**

MS in Data Science — University of Michigan-Dearborn

*Machine Learning · Customer Analytics · MLOps · Generative AI · Data Engineering*

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/rishignanasekar)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rishignanasekar)

</div>
