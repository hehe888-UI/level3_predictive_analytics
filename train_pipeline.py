"""
train_pipeline.py
------------------
Predictive Analytics & AI Model Deployment (Level 3 - Advanced)

Simple, beginner-friendly version. Steps:
1. Load the data and clean it.
2. Add a couple of simple new features (feature engineering).
3. Turn text categories (region, plan_type) into numbers using one-hot
   encoding (pd.get_dummies - a single function call).
4. Scale the numeric columns so they're all on a similar range.
5. Train three different models and compare them.
6. Save the best model so it can be used later for predictions.
7. Save some charts so we can see how the models performed.
"""

import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

base_folder = os.path.dirname(__file__)
data_file = os.path.join(base_folder, "data", "customer_data.csv")
model_folder = os.path.join(base_folder, "model")
output_folder = os.path.join(base_folder, "outputs")
os.makedirs(model_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

numeric_columns = ["age", "tenure_months", "monthly_spend", "num_purchases",
                    "support_tickets", "satisfaction_score",
                    "spend_per_purchase", "tickets_per_month"]
categorical_columns = ["region", "plan_type"]


# ---------- Step 1 & 2: Load data and add simple new features ----------
def load_and_prepare_data():
    df = pd.read_csv(data_file)
    df = df.drop_duplicates()
    df = df.dropna()

    # avoid dividing by zero
    safe_num_purchases = df["num_purchases"].replace(0, 1)
    safe_tenure = df["tenure_months"].replace(0, 1)

    df["spend_per_purchase"] = df["monthly_spend"] / safe_num_purchases
    df["tickets_per_month"] = df["support_tickets"] / safe_tenure

    return df


# ---------- Step 3 & 4: Turn categories into numbers, scale numeric columns ----------
def build_features(df, scaler=None, dummy_columns=None):
    # one-hot encode the categorical columns (e.g. region_North, plan_type_Basic, ...)
    df_with_dummies = pd.get_dummies(df, columns=categorical_columns)

    if dummy_columns is None:
        # first time: remember the column order/names so we can match it later
        dummy_columns = []
        for column in df_with_dummies.columns:
            if column not in numeric_columns and column not in ["customer_id", "churn"]:
                dummy_columns.append(column)
    else:
        # make sure new data has exactly the same dummy columns as training data
        for column in dummy_columns:
            if column not in df_with_dummies.columns:
                df_with_dummies[column] = 0

    feature_columns = numeric_columns + dummy_columns
    X = df_with_dummies[feature_columns].copy()

    if scaler is None:
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    else:
        X[numeric_columns] = scaler.transform(X[numeric_columns])

    return X, scaler, dummy_columns, feature_columns


# ---------- Step 5: Train and compare models ----------
def train_all_models(X_train, y_train):
    models = {}
    models["LogisticRegression"] = LogisticRegression(max_iter=1000)
    models["RandomForest"] = RandomForestClassifier(n_estimators=200, random_state=42)
    models["GradientBoosting"] = GradientBoostingClassifier(random_state=42)

    for model_name in models:
        models[model_name].fit(X_train, y_train)

    return models


def evaluate_all_models(models, X_test, y_test):
    results = {}
    roc_data = {}

    for model_name in models:
        model = models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        roc_data[model_name] = [fpr, tpr]

    return results, roc_data


def find_best_model_name(results):
    best_name = None
    best_score = -1
    for model_name in results:
        score = results[model_name]["roc_auc"]
        if score > best_score:
            best_score = score
            best_name = model_name
    return best_name


# ---------- Step 7: Charts / dashboards ----------
def make_charts(results, roc_data, models, X_test, y_test, df, best_model_name):
    # Chart 1: compare model scores
    metrics_table = pd.DataFrame(results).T
    metrics_table[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "model_comparison.png"), dpi=150)
    plt.close()

    # Chart 2: ROC curves
    plt.figure(figsize=(7, 6))
    for model_name in roc_data:
        fpr, tpr = roc_data[model_name]
        auc_score = results[model_name]["roc_auc"]
        plt.plot(fpr, tpr, label=model_name + " (AUC=" + str(round(auc_score, 2)) + ")")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_curves.png"), dpi=150)
    plt.close()

    # Chart 3: confusion matrix of the best model
    best_model = models[best_model_name]
    predictions = best_model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix - " + best_model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Chart 4: business dashboard - churn rate by plan type
    plt.figure(figsize=(7, 5))
    df.groupby("plan_type")["churn"].mean().plot(kind="bar", color="#C44E52")
    plt.ylabel("Churn Rate")
    plt.title("Churn Rate by Plan Type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "churn_by_plan.png"), dpi=150)
    plt.close()


# ---------- Main program ----------
def main():
    df = load_and_prepare_data()

    X, scaler, dummy_columns, feature_columns = build_features(df)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_all_models(X_train, y_train)
    results, roc_data = evaluate_all_models(models, X_test, y_test)
    best_model_name = find_best_model_name(results)

    make_charts(results, roc_data, models, X_test, y_test, df, best_model_name)

    # Save everything needed to make predictions later
    joblib.dump(models[best_model_name], os.path.join(model_folder, "churn_model.joblib"))
    joblib.dump(scaler, os.path.join(model_folder, "scaler.joblib"))
    joblib.dump(dummy_columns, os.path.join(model_folder, "dummy_columns.joblib"))
    joblib.dump(feature_columns, os.path.join(model_folder, "feature_columns.joblib"))

    report = {
        "results": results,
        "best_model": best_model_name,
        "feature_columns": feature_columns
    }
    with open(os.path.join(output_folder, "model_comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Model comparison:")
    for model_name in results:
        m = results[model_name]
        print(" ", model_name, ": accuracy=", round(m["accuracy"], 3),
              " precision=", round(m["precision"], 3),
              " recall=", round(m["recall"], 3),
              " f1=", round(m["f1"], 3),
              " roc_auc=", round(m["roc_auc"], 3))

    print("\nBest model:", best_model_name, "(saved to model/churn_model.joblib)")
    print("Charts saved to:", output_folder)


if __name__ == "__main__":
    main()
