import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor

from gate import should_promote
from inject_drift import make_drifted_prod

TARGET = "Global_active_power"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]
MODEL_NAME = "electricity-forecaster"
KIND = "shift"
DRIFT_START = pd.Timestamp("2010-04-30")
ALARM_END = pd.Timestamp("2010-05-07")   # first alarm (Step 48)
WAIT_DAYS, WINDOW_DAYS, SHADOW_DAYS = 28, 180, 28   # chosen from the Step 51 sweep (one scenario)
MIN_GAIN = 0.05

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("electricity-forecasting")
client = MlflowClient()


def fit(df):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    return model.fit(df[FEATURES], df[TARGET])


def abs_err(model, df):
    return (df[TARGET] - model.predict(df[FEATURES])).abs().values


def register(model, X, run_name, params, metrics, tags):
    """Log a model run to MLflow and register it as a new model version."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        sample = X.head(5).astype("float64")   # doubles, so MLflow does not warn about integer columns
        signature = infer_signature(sample, model.predict(sample))
        mlflow.sklearn.log_model(
            model, name="model", signature=signature, registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    for key, value in tags.items():           # also label the registered version itself
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    return version


def champion_version():
    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    except MlflowException:
        return None


read = lambda n: pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True)
train, val = read("train"), read("val")

# 1. Make sure a champion exists (first run only).
if champion_version() is None:
    first = fit(train)
    v = register(first, train[FEATURES], "initial_champion",
                 {"train_period": "2006-2008", "n_estimators": 100, "max_depth": 10},
                 {"train_rows": len(train)}, {"gate": "initial"})
    client.set_registered_model_alias(MODEL_NAME, "champion", v)
    print(f"No champion found: registered v{v} as champion.")

# 2. Load the current champion from the registry (this is also how serving will load it).
champ_v = champion_version()
champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Current champion: v{champ_v}")

# 3. Alarm happened at ALARM_END; wait for labels, retrain a challenger on recent data.
stream = make_drifted_prod(KIND, DRIFT_START)
history = pd.concat([train, val, stream])
t_retrain = ALARM_END + pd.Timedelta(days=WAIT_DAYS)
past = history[history.index < t_retrain]
recent = past[past.index >= t_retrain - pd.Timedelta(days=WINDOW_DAYS)]
challenger = fit(recent)

# 4. Shadow test: both models predict the next SHADOW_DAYS days; apply the gate.
shadow = stream[(stream.index >= t_retrain) & (stream.index < t_retrain + pd.Timedelta(days=SHADOW_DAYS))]
e_champ, e_chall = abs_err(champion, shadow), abs_err(challenger, shadow)
d = should_promote(e_champ, e_chall, min_gain=MIN_GAIN)
print(f"Shadow test ({len(shadow)} hours): champion MAE {e_champ.mean():.3f}, challenger MAE {e_chall.mean():.3f}, "
      f"gain {d['gain']:+.1%}, DM stat {d['dm_stat']:.2f}, p {d['p_value']:.4f}")

# 5. Register the challenger either way; move the champion alias only if the gate passes.
v = register(
    challenger, recent[FEATURES], f"challenger_{t_retrain.date()}",
    {"wait_days": WAIT_DAYS, "window_days": WINDOW_DAYS, "shadow_days": SHADOW_DAYS,
     "min_gain": MIN_GAIN, "drift_kind": KIND, "train_rows": len(recent)},
    {"champion_mae": float(e_champ.mean()), "challenger_mae": float(e_chall.mean()),
     "gain": d["gain"], "dm_stat": d["dm_stat"], "dm_p_value": d["p_value"]},
    {"gate": "promoted" if d["promote"] else "rejected", "compared_against": f"v{champ_v}"},
)
if d["promote"]:
    client.set_registered_model_alias(MODEL_NAME, "champion", v)
    print(f"PROMOTED: v{v} is now the champion (previous champion v{champ_v} kept for rollback).")
else:
    print(f"REJECTED: v{v} registered but champion stays v{champ_v}.")

print("\nRegistry now:")
current = champion_version()
for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
    tag = mv.tags.get("gate", "")
    print(f"  v{mv.version:<3} gate={tag:<9} {'<- champion' if int(mv.version) == current else ''}")