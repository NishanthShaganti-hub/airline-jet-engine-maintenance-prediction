
# Aircraft Engine Predictive Maintenance & Explainable AI System

## Overview

This project predicts the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS dataset and XGBoost machine learning. The system helps identify engine degradation early and supports proactive maintenance decisions.

## Features

* Remaining Useful Life (RUL) Prediction
* Feature Selection using Mutual Information
* XGBoost Regression Model
* Cross Validation
* Hyperparameter Tuning
* SHAP Explainability
* Learning Curve Analysis
* Streamlit Dashboard Deployment

## Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Joblib
* Streamlit

## Dataset

NASA CMAPSS Turbofan Engine Degradation Dataset

Target Variable:

```text
RUL = Maximum Cycle - Current Cycle
```

## Model Performance

| Metric   | Value |
| -------- | ----- |
| MAE      | 26.68 |
| RMSE     | 36.77 |
| R² Score | 0.704 |

## Workflow

```text
NASA CMAPSS Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
XGBoost Model
        ↓
Cross Validation
        ↓
Hyperparameter Tuning
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard
```

## Maintenance Categories

| Predicted RUL | Status   |
| ------------- | -------- |
| > 50          | SAFE     |
| 21 - 50       | WARNING  |
| ≤ 20          | CRITICAL |

## Run the Application

Train the model:

```bash
python src/train_model.py
```

Evaluate the model:

```bash
python src/evaluate_model.py
```

Launch Streamlit Dashboard:

```bash
python -m streamlit run src/app.py
```

## Project Structure

```text
data/
models/
src/
README.md
```

## Author

Shaganti Nishanth

B.Tech Information Technology

Kakatiya Institute of Technology and Science (KITSW)
=======
# airline-jet-engine-maintenance-prediction
>>>>>>> 2dd39f2889a5d89b3baece658074ed4563eec6e6
