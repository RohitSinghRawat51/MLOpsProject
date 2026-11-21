from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = "electricity-forecaster"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
state = {}


def load_champion():
    """Load whichever model version currently holds the 'champion' alias."""
    version = int(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    state["version"] = version


@asynccontextmanager
async def lifespan(app):
    load_champion()
    yield


app = FastAPI(title="Household electricity forecaster", lifespan=lifespan)


class Features(BaseModel):
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="0 = Monday")
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    Global_active_power_lag_1h: float = Field(ge=0)
    Global_active_power_lag_24h: float = Field(ge=0)
    Global_active_power_lag_168h: float = Field(ge=0)
    Global_active_power_roll_mean_24h: float = Field(ge=0)
    Global_active_power_roll_std_24h: float = Field(ge=0)


class ForecastRequest(BaseModel):
    timestamp: datetime = Field(description="The hour to forecast")
    last_168_hours: list[float] = Field(
        min_length=168, description="Hourly consumption (kW) for the 168 hours BEFORE `timestamp`, oldest first"
    )


def features_from_history(timestamp, history):
    """Build the model's features exactly as features.py does, from raw recent history."""
    h = np.asarray(history[-168:], dtype=float)
    ts = pd.Timestamp(timestamp)
    return pd.DataFrame([{
        "hour": ts.hour, "day_of_week": ts.dayofweek, "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "Global_active_power_lag_1h": h[-1],
        "Global_active_power_lag_24h": h[-24],
        "Global_active_power_lag_168h": h[-168],
        "Global_active_power_roll_mean_24h": h[-24:].mean(),
        "Global_active_power_roll_std_24h": h[-24:].std(ddof=1),
    }])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "champion_version": state["version"]}


@app.post("/predict")
def predict(features: Features):
    row = pd.DataFrame([features.model_dump()])[FEATURES]
    return {"prediction_kw": float(state["model"].predict(row)[0]), "model_version": state["version"]}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if not np.isfinite(req.last_168_hours).all():
        raise HTTPException(status_code=422, detail="history contains missing or non-finite values")
    row = features_from_history(req.timestamp, req.last_168_hours)
    return {
        "timestamp": req.timestamp.isoformat(),
        "prediction_kw": float(state["model"].predict(row)[0]),
        "model_version": state["version"],
    }


@app.post("/reload")
def reload_champion():
    """Pick up a newly promoted (or rolled-back) champion without restarting the service."""
    before = state["version"]
    load_champion()
    return {"previous_version": before, "champion_version": state["version"]}