from preprocessing import load_and_preprocess_data
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression
)

def select_features():

    df = load_and_preprocess_data()

    sensor_cols = [
        f"sensor_{i}"
        for i in range(1,22)
    ]

    X = df[sensor_cols]
    y = df["RUL"]

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=10
    )

    selector.fit(X, y)

    selected_features = X.columns[
        selector.get_support()
    ]

    return selected_features


if __name__ == "__main__":

    selected_features = select_features()

    print("\nTop 10 Selected Sensors:")
    print(selected_features)
    
    