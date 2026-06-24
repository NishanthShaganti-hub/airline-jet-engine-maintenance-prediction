from train_model import train_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import matplotlib.pyplot as plt
import os


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

    # ---------------------------------
    # Feature Importance
    # ---------------------------------

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

    # ---------------------------------
    # Create results folder
    # ---------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    # ---------------------------------
    # Plot Feature Importance
    # ---------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        importance["Feature"],
        importance["Importance"],
        color="#115E59",
        width=0.65
    )

    plt.title(
        "Feature Importance",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Features",
        fontsize=12
    )

    plt.ylabel(
        "Importance Score",
        fontsize=12
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nFeature Importance plot saved successfully!"
    )

    plt.show()


if __name__ == "__main__":

    evaluate_model()