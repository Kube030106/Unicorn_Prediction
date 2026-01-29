# Unicorn Prediction Code Explanation

This document explains the logic behind the `models/predict.py` script, which serves as the inference pipeline for the Unicorn Prediction project.

## Overview

The `predict.py` script is designed to:
1.  **Load** the trained machine learning model and data scaler.
2.  **Accept** new raw data for a startup.
3.  **Process** that data to match the format used during training.
4.  **Generate** a prediction (Success/Failure) and a probability score.

## Key Components

### 1. Environment Setup

```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

**Puppose:**
This block ensures that Python can correctly identify the project root directory. It resolves `ModuleNotFoundError` issues by dynamically adding the parent directory to the system path, regardless of where the script is run from.

### 2. Loading Artifacts

```python
model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
```

**Purpose:**
It loads the binary files saved during the training phase (`train.py`):
*   `best_model.pkl`: The trained classifier (e.g., Random Forest) that won the training selection.
*   `scaler.pkl`: The `StandardScaler` fitted on the training data. This is crucial to ensure new inputs are normalized exactly like the training data (e.g., scaling `funding_total_usd` correctly).

### 3. Feature Definition

**Purpose:**
The `FEATURES` list defines the exact specific order and selection of columns required by the model. This list must **exactly match** the features used in `train.py`. Any deviation here (missing columns or wrong order) would cause the model to crash or give incorrect predictions.

### 4. The Prediction Function (`predict_startup`)

This is the core logic function.

**Workflow:**
1.  **Input**: Accepts a dictionary (`input_data`) containing raw values (e.g., `{"funding_rounds": 4, "is_software": 1}`).
2.  **DataFrame Construction**: Converts the dictionary into a single-row pandas DataFrame.
3.  **Preprocessing**:
    *   `fillna(0)`: Handles potential missing values by replacing them with 0 (consistent with training).
    *   `scaler.transform()`: Scales numerical values using the loaded scaler.
4.  **Inference**:
    *   `model.predict()`: Returns the class label (`0` or `1`).
    *   `model.predict_proba()`: Returns the probability distribution. We extract the probability for class `1` (Success).

### 5. Output

The function returns a dictionary:
```python
{
    "is_success": 1,          # 1 = Likely IPO/Acquired, 0 = Likely Fail/Closed
    "probability": 0.812      # 81.2% confidence score
}
```
