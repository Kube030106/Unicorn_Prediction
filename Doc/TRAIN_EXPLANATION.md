# Training Pipeline Explanation (`train.py`)

This document explains the machine learning training pipeline used to build the Startup Success Predictor.

## 🎯 Goal
The script trains multiple machine learning models to classify startups as either **Success** (IPO/Acquired) or **Failure** (Closed/Operating but not exited). It automatically selects the best-performing model based on the **F1 Score** and saves it for future use.

## 🛠️ Key Steps

### 1. Data Loading & Cleaning
*   **Source**: Loads `datasets/startupdata.csv`.
*   **Target Definitions**:
    *   **Success (1)**: Status is `ipo` or `acquired`.
    *   **Failure (0)**: Anything else (e.g., `closed`).
*   **Missing Values**: Fills missing numerical values with `0`.

### 2. Feature Selection
The model uses a mix of financial, temporal, and categorical signals. See [Features.md](Features.md) for a deep dive into each signal.

### 3. Preprocessing
*   **Stratified Split**: Splits data into 80% Training and 20% Testing.
    *   *Why Stratified?* To ensure the proportion of successful startups is the same in both sets.
*   **Standard Scaling**: Normalizes all numerical features (e.g., `funding_total_usd`) so that large numbers don't dominate the model.

### 4. Handling Imbalance
Startups naturally have a low success rate, creating "Class Imbalance".
*   **Solution**: We calculate `scale_pos_weight` (Ratio of Negative to Positive cases) and pass it to **XGBoost** and use `class_weight="balanced"` for **Random Forest** and **Logistic Regression**. This forces the models to pay more attention to the rare "Success" cases.

### 5. Model Competition
We train four different algorithms to see which works best:

| Model | Strengths | Configuration |
| :--- | :--- | :--- |
| **Logistic Regression** | Simple, interpretable baseline. | Balanced class weights. |
| **Random Forest** | Robust to noise, handles non-linear data well. | 300 Trees, Max Depth 12. |
| **Gradient Boosting** | High accuracy, builds trees sequentially. | Default settings. |
| **XGBoost** | The industry standard for tabular data. Optimized for speed and performance. | Learning rate 0.05, Subsample 0.8. |

### 6. Evaluation & Saving
*   **Metric**: We use **F1 Score** (harmonic mean of precision and recall) instead of just Accuracy.
    *   *Why?* In imbalanced datasets, Accuracy can be misleading (a model that predicts "Fail" for everyone would still have high accuracy). F1 ensures we actually catch the successful companies.
*   **Winner Selection**: The script compares the F1 scores of all 4 models and picks the winner.
*   **Artifacts**:
    *   `artifacts/best_model.pkl`: The winning model file.
    *   `artifacts/scaler.pkl`: The scaler used (must be saved to process new data later).

## 🚀 How to Run
```bash
python models/train.py
```
This will print the results and save the best model to the `artifacts/` folder.
