
from train_model import train_model


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import matplotlib.pyplot as plt

def classify_risk(rul):

    if rul > 50:
        return "SAFE"

    elif rul > 20:
        return "WARNING"

    else:
        return "CRITICAL"


def evaluate_model():

    model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nModel Performance")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    results = pd.DataFrame({
        "Actual_RUL": y_test,
        "Predicted_RUL": y_pred
    })

    results["Status"] = (
        results["Predicted_RUL"]
        .apply(classify_risk)
    )

    print("\nSample Predictions:")
    print(results.head(20))
    
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance)

    plt.figure(figsize=(8,5))

    plt.bar(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.show()


if __name__ == "__main__":

    evaluate_model()