from preprocessing import load_and_preprocess_data
from feature_selection import select_features

from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor

import numpy as np
import matplotlib.pyplot as plt

# Load data
df = load_and_preprocess_data()

selected_features = select_features()

X = df[list(selected_features) + ["cycle"]]
y = df["RUL"]

# Model
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Mean scores
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    train_sizes,
    train_mean,
    label="Training Score"
)

plt.plot(
    train_sizes,
    val_mean,
    label="Validation Score"
)

plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve")

plt.legend()

plt.show()