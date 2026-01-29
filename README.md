# 🦄 Unicorn Prediction: Startup Success Classifier

A Machine Learning project that predicts whether a startup will become successful (IPO or Acquired) based on funding, location, industry, and network signals.

## 🚀 Overview
This project uses historical startup data to build a predictive model. It analyzes varied signals—from funding velocity to investor quality—to estimate the probability of a "successful exit".

**Key Goals:**
*   Analyze key drivers of startup success.
*   Train multiple classifiers (Random Forest, XGBoost, etc.).
*   Deploy a simple inference script for new predictions.

## 📂 Project Structure

```bash
UnicornPrediction/
├── artifacts/             # 🧠 Saved Models (Generated via training)
│   ├── best_model.pkl     # The winning classifier (Random Forest/XGBoost)
│   └── scaler.pkl         # StandardScaler for normalizing new inputs
│
├── datasets/              
│   └── startupdata.csv    # 📊 Raw historical data (Input)
│
├── Doc/                   # 📚 Project Documentation
│   ├── Features.md        # Deep dive into the 24 predictive signals
│   ├── TRAIN_EXPLANATION.md   # Logic behind the training pipeline
│   └── PREDICT_EXPLANATION.md # How the inference script works
│
├── features/              
│   └── feature_engineering.py # 🛠️ Logic to clean & transform raw data
│
├── models/                # 🤖 Machine Learning Core
│   ├── train.py           # Script to train, evaluate & save models
│   └── predict.py         # Script to predict success for new startups
│
├── requirements.txt       # 📦 Dependencies (pandas, sklearn, joblib)
└── README.md              # 🏠 Project Overview (You are here)
```

## 🛠️ Installation

1.  **Clone the repository** (or navigate to the folder).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Model Training

To retrain the models and find the best one:

```bash
python models/train.py
```

**What happens?**
*   Loads data from `datasets/startupdata.csv`.
*   Cleans and scales features.
*   Trains 4 models: **Logistic Regression, Random Forest, Gradient Boosting, XGBoost**.
*   Selects the best model based on **F1 Score**.
*   Saves the winner to `artifacts/best_model.pkl`.

## 🔮 making Predictions

To test the model on a sample startup or integrate it:

```bash
python models/predict.py
```

It takes a dictionary of startup details (funding, location, industry) and outputs:
*   **Result**: 1 (Success) / 0 (Fail)
*   **Probablity**: Confidence score (e.g., 0.812).

## 📊 Features Used
The model relies on ~24 key features, including:
*   **Funding**: Total USD, Number of Rounds.
*   **Speed**: Time to first funding, round continuity.
*   **Network**: Investor count, presence of VCs/Angels.
*   **Ecosystem**: Location (CA, NY, etc.) and Industry (Software, Web, etc.).

See [Doc/Features.md](Doc/Features.md) for a deep dive.