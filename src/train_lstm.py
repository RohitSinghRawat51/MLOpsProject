import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TensorFlow's C++ startup logs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

TARGET = "Global_active_power"
WINDOW = 24  # use the past 24 hours to predict the next one


def make_sequences(df, target=TARGET, window=WINDOW):
    """Build (X, y) sequences, skipping any window that crosses a gap in the
    hourly timestamps (e.g. a meter outage removed during feature engineering)."""
    values = df[target].values
    timestamps = df.index
    X, y = [], []
    for i in range(len(df) - window):
        window_idx = timestamps[i : i + window + 1]
        gaps = window_idx.to_series().diff().dropna()
        if (gaps == pd.Timedelta(hours=1)).all():
            X.append(values[i : i + window])
            y.append(values[i + window])
    return np.array(X), np.array(y)


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    val = pd.read_csv("data/val.csv", index_col=0, parse_dates=True)

    # Scale using train data only
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    train_scaled[TARGET] = scaler.fit_transform(train[[TARGET]])
    val_scaled = val.copy()
    val_scaled[TARGET] = scaler.transform(val[[TARGET]])

    X_train, y_train = make_sequences(train_scaled)
    X_val, y_val = make_sequences(val_scaled)

    X_train = X_train.reshape(-1, WINDOW, 1)
    X_val = X_val.reshape(-1, WINDOW, 1)

    print(f"Train sequences: {X_train.shape[0]} (window={WINDOW}h)")
    print(f"Validation sequences: {X_val.shape[0]}")

    mlflow.set_experiment("electricity-forecasting")

    with mlflow.start_run(run_name="lstm_baseline"):
        params = {"window": WINDOW, "lstm_units": 32, "epochs": 10, "batch_size": 32}
        mlflow.log_params(params)

        model = keras.Sequential([
            keras.layers.Input(shape=(WINDOW, 1)),
            keras.layers.LSTM(params["lstm_units"]),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_data=(X_val, y_val),
            verbose=1,
        )

        y_pred_scaled = model.predict(X_val).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        scores = evaluate(y_true, y_pred)
        mlflow.log_metrics(scores)

        signature = infer_signature(X_val, y_pred_scaled.reshape(-1, 1))
        mlflow.tensorflow.log_model(model, name="model", signature=signature)

        print("\nLSTM on validation set:")
        print(f"  MAE:  {scores['MAE']:.4f}")
        print(f"  RMSE: {scores['RMSE']:.4f}")
        print(f"\nCompare to XGBoost: MAE 0.3610, RMSE 0.5257")
        print(f"Compare to Random Forest: MAE 0.3590, RMSE 0.5254")
        print(f"Compare to seasonal-naive baseline: MAE 0.5656, RMSE 0.8452")