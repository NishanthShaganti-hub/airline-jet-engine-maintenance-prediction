from preprocessing import load_and_preprocess_data
from feature_selection import select_features

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

import numpy as np

df = load_and_preprocess_data()

selected_features = select_features()

X = df[list(selected_features) + ["cycle"]]
y = df["RUL"]

model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation R² Scores:")
print(scores)

print("\nAverage R²:")
print(np.mean(scores))

print("\nStandard Deviation:")
print(np.std(scores))