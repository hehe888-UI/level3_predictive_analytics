# Predictive Analytics & AI Model Deployment (Level 3 - Advanced)
`cognevance_predictive-analytics-deployment`

## Overview
A complete ML pipeline that predicts **customer churn** from account and
usage data: preprocessing → feature engineering → training multiple
models → comparison → deployment via a Flask REST API → dashboards.

## Workflow
1. **Data collection** — `generate_dataset.py` creates
   `data/customer_data.csv`, a synthetic business/customer dataset.
   Replace with a real business/healthcare/finance/sales dataset (e.g.
   Kaggle's Telco Customer Churn) for the real submission — keep or
   adapt the column names used in `train_pipeline.py`.
2. **Preprocessing & feature engineering** — duplicates/nulls dropped;
   engineered features `spend_per_purchase` and `tickets_per_month`
   added; numeric features scaled and categorical features one-hot
   encoded via a `ColumnTransformer` (`train_pipeline.py`).
3. **Model training** — three models trained: Logistic Regression,
   Random Forest, Gradient Boosting (`train_models`).
4. **Comparison** — accuracy, precision, recall, F1, and ROC-AUC computed
   for each model (`evaluate`); the best model (by ROC-AUC) is selected
   and saved with `joblib` for deployment.
5. **Deployment** — `deploy_api.py` is a Flask REST API exposing
   `/predict` (churn prediction) and `/health`.
6. **Dashboards** — model comparison bar chart, ROC curves, confusion
   matrix for the best model, and a business dashboard (churn rate by
   plan type), all saved as PNGs.
7. **Documentation** — this README + `outputs/model_comparison_report.json`
   documents the full pipeline architecture and results.

## Technologies Used
Python, pandas, scikit-learn (Logistic Regression, Random Forest,
Gradient Boosting, StandardScaler), matplotlib, Flask, joblib

## A note on the code style
Written with plain functions instead of classes, and without scikit-learn's
`Pipeline`/`ColumnTransformer` (which are a bit more advanced). Instead,
`build_features()` does encoding and scaling as clear, separate steps
using `pd.get_dummies()` and `StandardScaler` directly, so you can see
exactly what happens to the data at each stage.

## How to Run
```bash
pip install -r requirements.txt

python generate_dataset.py   # creates data/customer_data.csv
python train_pipeline.py     # trains, compares, saves model + dashboards

python deploy_api.py         # serves the model at http://127.0.0.1:5000
```

### Example prediction request
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 34, "region": "North", "plan_type": "Standard",
        "tenure_months": 12, "monthly_spend": 85.5,
        "num_purchases": 5, "support_tickets": 2,
        "satisfaction_score": 6
      }'
```

## Outputs
- `model/churn_model.joblib`, `model/feature_cols.joblib` — deployable model
- `outputs/model_comparison.png` — accuracy/precision/recall/F1/ROC-AUC bar chart
- `outputs/roc_curves.png` — ROC curve per model
- `outputs/confusion_matrix.png` — confusion matrix for the best model
- `outputs/churn_by_plan.png` — business dashboard (churn rate by plan)
- `outputs/model_comparison_report.json` — full metrics + pipeline info

## Project Structure
```
level3_predictive_analytics/
├── generate_dataset.py
├── train_pipeline.py
├── deploy_api.py
├── requirements.txt
├── data/
│   └── customer_data.csv
├── model/
│   ├── churn_model.joblib
│   ├── scaler.joblib
│   ├── dummy_columns.joblib
│   └── feature_columns.joblib
└── outputs/
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── churn_by_plan.png
    └── model_comparison_report.json
```

## Extending This Project
- Swap the synthetic dataset for a real one (Kaggle/business data).
- Add a deep learning model (Keras/TensorFlow) to the comparison.
- Containerize `deploy_api.py` with Docker, or deploy to a cloud platform
  (AWS/GCP/Azure/Render) for the "cloud platform" deployment option.
- Build a richer dashboard (e.g. Streamlit) on top of the saved model.
