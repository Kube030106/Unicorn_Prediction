import sys
import os

# -----------------------------
# Fix project root import
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import joblib

# -----------------------------
# Load model & scaler
# -----------------------------
model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

FEATURES = [
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "age_first_funding_year",
    "age_last_funding_year",
    "relationships",
    "avg_participants",
    "has_VC",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
    "is_CA",
    "is_NY",
    "is_MA",
    "is_TX",
    "is_software",
    "is_web",
    "is_mobile",
    "is_enterprise",
    "is_ecommerce",
    "is_biotech",
    "is_consulting"
]

def predict_startup(input_data: dict):
    df = pd.DataFrame([input_data])
    df = df[FEATURES].fillna(0)

    X = scaler.transform(df)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "is_success": int(prediction),
        "probability": round(probability, 3)
    }

# -----------------------------
# Example usage
# -----------------------------
# Success Scenario
# sample_startup = {
#     "funding_rounds": 4,
#     "funding_total_usd": 120_000_000,
#     "milestones": 6,
#     "age_first_funding_year": 1.2,
#     "age_last_funding_year": 4.5,
#     "relationships": 10,
#     "avg_participants": 3,
#     "has_VC": 1,
#     "has_angel": 1,
#     "has_roundA": 1,
#     "has_roundB": 1,
#     "has_roundC": 0,
#     "has_roundD": 0,
#     "is_CA": 0,
#     "is_NY": 0,
#     "is_MA": 0,
#     "is_TX": 0,
#     "is_software": 1,
#     "is_web": 0,
#     "is_mobile": 1,
#     "is_enterprise": 0,
#     "is_ecommerce": 0,
#     "is_biotech": 0,
#     "is_consulting": 0
# }
# Failure Scenario
sample_startup = {
    "funding_rounds": 1,
    "funding_total_usd": 300_000,      # Very low funding
    "milestones": 1,
    "age_first_funding_year": 4.5,     # Very slow first funding
    "age_last_funding_year": 5.0,
    "relationships": 1,
    "avg_participants": 1,

    "has_VC": 0,
    "has_angel": 1,
    "has_roundA": 0,
    "has_roundB": 0,
    "has_roundC": 0,
    "has_roundD": 0,

    "is_CA": 0,
    "is_NY": 0,
    "is_MA": 0,
    "is_TX": 0,

    "is_software": 0,
    "is_web": 1,
    "is_mobile": 0,
    "is_enterprise": 0,
    "is_ecommerce": 0,
    "is_biotech": 0,
    "is_consulting": 1
}


result = predict_startup(sample_startup)
print(result)
