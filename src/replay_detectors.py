import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor

from drift import calculate_psi
from inject_drift import make_drifted_prod

TARGET = "Global_active_power"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]
WINDOW = pd.Timedelta(days=7)
MIN_HOURS = 100                       # skip weeks with too little data (outages)
DRIFT_START = pd.Timestamp("2010-04-30")   # lines up with a weekly window boundary
MAGNITUDE = 0.5

read = lambda n: pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True)
train, val, prod = read("train"), read("val"), read("prod")

# The champion model: trained once on 2006-2008, never retrained in this script.
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(train[FEATURES], train[TARGET])

all_ref = train[TARGET].values
month_ref = {m: train[train.index.month == m][TARGET].values for m in range(1, 13)}


def score_frame(df):
    """Score every non-overlapping 7-day window of df with each drift signal."""
    rows, t = [], df.index.min().normalize()
    while t < df.index.max():
        w = df.loc[t: t + WINDOW - pd.Timedelta(hours=1)]
        if len(w) >= MIN_HOURS:
            y = w[TARGET].values
            ref_m = month_ref[(t + WINDOW / 2).month]
            ks = ks_2samp(ref_m, y)
            rows.append({
                "start": t, "end": t + WINDOW,
                "psi_all": calculate_psi(all_ref, y),
                "psi_month": calculate_psi(ref_m, y),
                "ks_month": ks.statistic, "ks_p": ks.pvalue,
                "mae": np.mean(np.abs(y - model.predict(w[FEATURES]))),
            })
        t += WINDOW
    return pd.DataFrame(rows).set_index("start")


# Thresholds calibrated on the drift-free validation year (2009): the largest
# value each signal reached on normal data.
sf_val = score_frame(val)
thr = sf_val[["psi_month", "ks_month", "mae"]].max()
print("Calibrated on 2009 weekly windows (max of normal variation):")
print(thr.round(3).to_string(), "\n")


def alarms(sf):
    return pd.DataFrame({
        "PSI all-season > 0.25 (textbook)": sf["psi_all"] > 0.25,
        "KS p < 0.05 (textbook)": sf["ks_p"] < 0.05,
        "PSI same-month (calibrated)": sf["psi_month"] > thr["psi_month"],
        "KS same-month (calibrated)": sf["ks_month"] > thr["ks_month"],
        "Error-based (calibrated)": sf["mae"] > thr["mae"],
    })


def report(title, sf, drift_start=None):
    a = alarms(sf)
    print(f"=== {title} ({len(sf)} weekly windows) ===")
    rows = []
    for det in a.columns:
        hit = a[det]
        if drift_start is None:
            rows.append({"detector": det, "false alarms": f"{int(hit.sum())}/{len(hit)}"})
            continue
        pre, post = sf["end"] <= drift_start, sf.index >= drift_start
        first = sf.loc[post & hit, "end"].min()
        rows.append({
            "detector": det,
            "false alarms pre-drift": f"{int(hit[pre].sum())}/{int(pre.sum())}",
            "delay (days)": "missed" if pd.isna(first) else (first - drift_start).days,
            "post-drift weeks flagged": f"{int(hit[post].sum())}/{int(post.sum())}",
        })
    print(pd.DataFrame(rows).to_string(index=False), "\n")


report("No drift (control)", score_frame(prod))
for kind in ("sudden", "gradual", "recurring"):
    stream = make_drifted_prod(kind, DRIFT_START, magnitude=MAGNITUDE)
    report(f"{kind} drift, +{int(MAGNITUDE * 100)}% from {DRIFT_START.date()}", score_frame(stream), DRIFT_START)