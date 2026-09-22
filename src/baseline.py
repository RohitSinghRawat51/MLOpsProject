import numpy as np
import pandas as pd

TARGET = "Global_active_power"


def seasonal_naive_predict(df, target=TARGET, lag_col="Global_active_power_lag_168h"):
    """Predict this hour's value as the same hour exactly one week (168h) ago."""
    return df[lag_col]


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    val = pd.read_csv("data/val.csv", index_col=0, parse_dates=True)

    y_true = val[TARGET]
    y_pred = seasonal_naive_predict(val)

    scores = evaluate(y_true, y_pred)
    print("Seasonal-naive baseline on validation set:")
    print(f"  MAE:  {scores['MAE']:.4f}")
    print(f"  RMSE: {scores['RMSE']:.4f}")