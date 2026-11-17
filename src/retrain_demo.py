import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from inject_drift import make_drifted_prod

TARGET = "Global_active_power"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]
DRIFT_START = pd.Timestamp("2010-04-30")
ALARM_END = pd.Timestamp("2010-05-07")   # first window the error trigger flagged (Step 48)
SHADOW_DAYS = 28                          # how long the challenger is tested before a decision
KIND = "shift"                            # "sudden" (scale-up) or "shift" (daily pattern moves 3h later)
MAGNITUDE = 0.5                           # for "sudden"; set to 0 for a no-drift control run
MIN_GAIN = 0.05                           # challenger must also be at least 5% better, not just significant

read = lambda n: pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True)
train, val = read("train"), read("val")
stream = make_drifted_prod(KIND, DRIFT_START, **({"magnitude": MAGNITUDE} if KIND == "sudden" else {}))
history = pd.concat([train, val, stream])     # everything that exists, in time order


def fit(df):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    return model.fit(df[FEATURES], df[TARGET])


def abs_err(model, df):
    return (df[TARGET] - model.predict(df[FEATURES])).abs().values


def diebold_mariano(err_champion, err_challenger, lags=24):
    """Diebold-Mariano test on absolute-error loss.

    d_t = champion loss - challenger loss. A positive statistic means the
    challenger is better. Hourly errors are autocorrelated, so the variance
    uses a Bartlett-weighted (HAC) estimate over `lags` lags, not the plain one.
    """
    d = np.asarray(err_champion) - np.asarray(err_challenger)
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:                              # identical models: nothing to test
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / n)
    return stat, 2 * (1 - norm.cdf(abs(stat)))


champion = fit(train)                         # trained once on 2006-2008
rows = []
for wait_days in (7, 28):                     # label-collection time after the alarm
    t_retrain = ALARM_END + pd.Timedelta(days=wait_days)
    shadow = stream[(stream.index >= t_retrain) & (stream.index < t_retrain + pd.Timedelta(days=SHADOW_DAYS))]
    past = history[history.index < t_retrain]
    for label, days in (("all history", None), ("last 180 days", 180), ("last 60 days", 60)):
        data = past if days is None else past[past.index >= t_retrain - pd.Timedelta(days=days)]
        challenger = fit(data)
        e_champ, e_chall = abs_err(champion, shadow), abs_err(challenger, shadow)
        stat, p = diebold_mariano(e_champ, e_chall)
        gain = 1 - e_chall.mean() / e_champ.mean()
        rows.append({
            "wait (days)": wait_days, "training data": label, "train rows": len(data),
            "champion MAE": round(e_champ.mean(), 3), "champ MAE / mean load": round(e_champ.mean() / shadow[TARGET].mean(), 3),
            "challenger MAE": round(e_chall.mean(), 3), "gain": f"{gain:+.1%}",
            "DM stat": round(stat, 2), "p-value": round(p, 4),
            "promote?": "YES" if (stat > 0 and p < 0.05 and gain >= MIN_GAIN) else "no",
        })
print(f"Shadow test on the {SHADOW_DAYS} days after retraining ({KIND} drift from {DRIFT_START.date()}, magnitude {MAGNITUDE})\n")
print(pd.DataFrame(rows).to_string(index=False))