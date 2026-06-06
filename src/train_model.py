from preprocessing import load_and_preprocess_data
from feature_selection import select_features

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import joblib


def train_model():

    df = load_and_preprocess_data()

    selected_features = select_features()

    X = df[list(selected_features) + ["cycle"]]
    y = df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "models/xgb_rul_model.pkl"
    )

    return model, X_test, y_test


if __name__ == "__main__":

    model, X_test, y_test = train_model()

    print("Model trained successfully!")
    print("Testing Shape:", X_test.shape)