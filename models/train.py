# import sys
# import os
# import joblib
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# # -----------------------------
# # Fix import path (Windows-safe)
# # -----------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, PROJECT_ROOT)

# # -----------------------------
# # Load dataset
# # -----------------------------
# df = pd.read_csv("datasets/startupdata.csv")

# # -----------------------------
# # Clean column names
# # -----------------------------
# df.columns = df.columns.str.strip()

# # -----------------------------
# # Create target variable
# # -----------------------------
# df["is_success"] = df["status"].apply(
#     lambda x: 1 if x in ["ipo", "acquired"] else 0
# )

# # -----------------------------
# # Feature selection
# # -----------------------------
# FEATURES = [
#     "funding_rounds",
#     "funding_total_usd",
#     "milestones",
#     "age_first_funding_year",
#     "age_last_funding_year",
#     "relationships",
#     "avg_participants",

#     "has_VC",
#     "has_angel",
#     "has_roundA",
#     "has_roundB",
#     "has_roundC",
#     "has_roundD",

#     "is_CA",
#     "is_NY",
#     "is_MA",
#     "is_TX",

#     "is_software",
#     "is_web",
#     "is_mobile",
#     "is_enterprise",
#     "is_ecommerce",
#     "is_biotech",
#     "is_consulting"
# ]

# # Fill missing values
# df[FEATURES] = df[FEATURES].fillna(0)

# X = df[FEATURES]
# y = df["is_success"]

# # -----------------------------
# # Train / Test split
# # -----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # -----------------------------
# # Scaling
# # -----------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # -----------------------------
# # Models (primary + fallback)
# # -----------------------------
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=1000),
#     "RandomForest": RandomForestClassifier(
#         n_estimators=300,
#         max_depth=12,
#         random_state=42
#     ),
#     "GradientBoosting": GradientBoostingClassifier(random_state=42),
    
# }

# best_model = None
# best_f1 = 0
# best_name = ""

# # -----------------------------
# # Training & evaluation
# # -----------------------------
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     preds = model.predict(X_test_scaled)

#     acc = accuracy_score(y_test, preds)
#     f1 = f1_score(y_test, preds)

#     print(f"{name} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

#     if f1 > best_f1:
#         best_f1 = f1
#         best_model = model
#         best_name = name

# print(f"\n🏆 Best Model: {best_name} (F1 = {best_f1:.4f})")

# # -----------------------------
# # Save artifacts
# # -----------------------------
# os.makedirs("artifacts", exist_ok=True)
# joblib.dump(best_model, "artifacts/best_model.pkl")
# joblib.dump(scaler, "artifacts/scaler.pkl")

import sys
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# Fix project root import
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("datasets/startupdata.csv")
df.columns = df.columns.str.strip()

# --------------------------------------------------
# Target variable (SUCCESS)
# --------------------------------------------------
df["is_success"] = df["status"].apply(
    lambda x: 1 if x in ["ipo", "acquired"] else 0
)

# --------------------------------------------------
# Feature list (UNCHANGED)
# --------------------------------------------------
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

df[FEATURES] = df[FEATURES].fillna(0)

X = df[FEATURES]
y = df["is_success"]

# --------------------------------------------------
# Train-test split (stratified)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Handle class imbalance for XGBoost
# --------------------------------------------------
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# --------------------------------------------------
# Models (XGBoost = primary)
# --------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_f1 = 0
best_name = ""

# --------------------------------------------------
# Train & evaluate
# --------------------------------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"{name} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print(f"\n🏆 Best Model Selected: {best_name} (F1 = {best_f1:.4f})")

# --------------------------------------------------
# Save artifacts
# --------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model, "artifacts/best_model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
