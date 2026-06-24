import shap
import joblib
import matplotlib.pyplot as plt
import os

from preprocessing import load_and_preprocess_data
from feature_selection import select_features



# Loading data
df = load_and_preprocess_data()

selected_features = select_features()

X = df[list(selected_features) + ["cycle"]]

# Load trained model
model = joblib.load(
    "models/xgb_rul_model.pkl"
)

# Creating SHAP explainer
explainer = shap.TreeExplainer(model)

# Small sample for faster computation
X_sample = X.sample(
    100,
    random_state=42
)

# Calculating SHAP values
shap_values = explainer.shap_values(
    X_sample
)

# SHAP Summary Plot
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

# Save Image
plt.savefig(
    "results/shap_summary.png",
    bbox_inches="tight"
)

print(
    "SHAP Summary Plot saved successfully!"
)

plt.show()