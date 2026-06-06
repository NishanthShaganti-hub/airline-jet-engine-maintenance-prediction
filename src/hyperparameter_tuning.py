from preprocessing import load_and_preprocess_data
from feature_selection import select_features

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

df = load_and_preprocess_data()

selected_features = select_features()

X = df[list(selected_features) + ["cycle"]]
y = df["RUL"]

model = XGBRegressor(
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X, y)

print("Best Parameters:")
print(grid.best_params_)

print("\nBest R²:")
print(grid.best_score_)