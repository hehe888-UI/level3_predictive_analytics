"""
deploy_api.py
-------------
Simple Flask REST API that serves the trained churn-prediction model.

Run with:
    python deploy_api.py
Then send a POST request to http://127.0.0.1:5000/predict, for example:

    curl -X POST http://127.0.0.1:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
            "age": 34, "region": "North", "plan_type": "Standard",
            "tenure_months": 12, "monthly_spend": 85.5,
            "num_purchases": 5, "support_tickets": 2,
            "satisfaction_score": 6
          }'
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

base_folder = os.path.dirname(__file__)
model_folder = os.path.join(base_folder, "model")

app = Flask(__name__)

# Load everything we saved during training
model = joblib.load(os.path.join(model_folder, "churn_model.joblib"))
scaler = joblib.load(os.path.join(model_folder, "scaler.joblib"))
dummy_columns = joblib.load(os.path.join(model_folder, "dummy_columns.joblib"))
feature_columns = joblib.load(os.path.join(model_folder, "feature_columns.joblib"))

numeric_columns = ["age", "tenure_months", "monthly_spend", "num_purchases",
                    "support_tickets", "satisfaction_score",
                    "spend_per_purchase", "tickets_per_month"]

required_fields = ["age", "region", "plan_type", "tenure_months", "monthly_spend",
                    "num_purchases", "support_tickets", "satisfaction_score"]


def prepare_single_input(data):
    """Turns one JSON request into a row of features the model understands,
    using the exact same steps used during training."""

    row = {}
    for field in required_fields:
        row[field] = data[field]

    # feature engineering - same formulas as train_pipeline.py
    num_purchases = row["num_purchases"] if row["num_purchases"] != 0 else 1
    tenure_months = row["tenure_months"] if row["tenure_months"] != 0 else 1
    row["spend_per_purchase"] = row["monthly_spend"] / num_purchases
    row["tickets_per_month"] = row["support_tickets"] / tenure_months

    df = pd.DataFrame([row])

    # one-hot encode region/plan_type the same way as training
    df = pd.get_dummies(df, columns=["region", "plan_type"])
    for column in dummy_columns:
        if column not in df.columns:
            df[column] = 0

    X = df[feature_columns].copy()
    X[numeric_columns] = scaler.transform(X[numeric_columns])
    return X


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if len(missing_fields) > 0:
        return jsonify({"error": "Missing required fields: " + str(missing_fields)}), 400

    try:
        X = prepare_single_input(data)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
    except Exception as error:
        return jsonify({"error": str(error)}), 400

    if prediction == 1:
        label = "Churn"
    else:
        label = "No Churn"

    result = {
        "churn_prediction": prediction,
        "churn_label": label,
        "churn_probability": round(probability, 4)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
