import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TARGET = "Global_active_power"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    val = pd.read_csv("data/val.csv", index_col=0, parse_dates=True)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    mlflow.set_experiment("electricity-forecasting")

    with mlflow.start_run(run_name="random_forest_baseline"):
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)

        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        scores = evaluate(y_val, y_pred)
        mlflow.log_metrics(scores)

        mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
)

        print("Random Forest on validation set:")
        print(f"  MAE:  {scores['MAE']:.4f}")
        print(f"  RMSE: {scores['RMSE']:.4f}")
        print(f"\nCompare to seasonal-naive baseline: MAE 0.5656, RMSE 0.8452")