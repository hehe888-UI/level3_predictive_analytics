"""
generate_dataset.py
--------------------
Creates a simple customer dataset (customer_data.csv) for predicting
customer churn (whether a customer will leave / stop using the service).

NOTE: For the real submission, replace this with a real business dataset
(for example, Kaggle's "Telco Customer Churn" dataset). This script just
creates sample data so the project runs end-to-end.
"""

import random
import math
import pandas as pd
import os

random.seed(7)

output_folder = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(output_folder, exist_ok=True)

number_of_customers = 2000
region_list = ["North", "South", "East", "West"]
plan_list = ["Basic", "Standard", "Premium"]

rows = []

for i in range(number_of_customers):
    customer_id = i + 1
    age = random.randint(18, 70)
    region = random.choice(region_list)
    plan_type = random.choices(plan_list, weights=[0.4, 0.4, 0.2])[0]
    tenure_months = random.randint(1, 72)
    monthly_spend = round(random.gammavariate(4, 25), 2)
    num_purchases = max(0, round(random.gauss(6, 3)))
    support_tickets = max(0, round(random.gauss(1.2, 1.2)))
    satisfaction_score = random.randint(1, 10)

    # Decide churn using a simple rule + some randomness, so the data has
    # a real pattern the models can learn (customers with low satisfaction,
    # a "Basic" plan, and many support tickets churn more often).
    churn_score = 0.0
    churn_score += support_tickets * 0.15
    churn_score -= (satisfaction_score - 7) * 0.15
    churn_score -= (tenure_months - 30) * 0.01
    churn_score -= (monthly_spend - 100) * 0.004

    if plan_type == "Basic":
        churn_score += 0.15
    if plan_type == "Premium":
        churn_score -= 0.2

    churn_score += random.gauss(0, 0.5)

    # turn the score into a probability between 0 and 1
    churn_probability = 1 / (1 + math.exp(-(churn_score - 1.2)))
    is_churn = 1 if random.random() < churn_probability else 0

    rows.append([
        customer_id, age, region, plan_type, tenure_months, monthly_spend,
        num_purchases, support_tickets, satisfaction_score, is_churn
    ])

columns = ["customer_id", "age", "region", "plan_type", "tenure_months",
           "monthly_spend", "num_purchases", "support_tickets",
           "satisfaction_score", "churn"]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(os.path.join(output_folder, "customer_data.csv"), index=False)

churn_rate = df["churn"].mean()
print("Generated", len(df), "rows ->", os.path.join(output_folder, "customer_data.csv"))
print("Churn rate:", round(churn_rate * 100, 2), "%")
