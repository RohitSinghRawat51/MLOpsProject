import numpy as np
import pandas as pd

from features import build_feature_table
from split import split_data

TARGET = "Global_active_power"


def inject_drift(df, kind, start, magnitude=0.5, ramp_days=60, period_days=60):
    """Return a copy of df with synthetic drift applied to TARGET from `start`.

    df must have the full hourly index (gaps left as NaN), so that lag and
    rolling features can be rebuilt correctly afterwards.

    kind:
      sudden      multiply consumption by (1 + magnitude) from `start` on
      gradual     same, but the multiplier ramps in linearly over ramp_days
      recurring   the shift is on for period_days, off for period_days, repeating
    """
    out = df.copy()
    start = pd.Timestamp(start)
    days = np.asarray((out.index - start).total_seconds() / 86400)
    if kind == "sudden":
        factor = np.where(days >= 0, 1 + magnitude, 1.0)
    elif kind == "gradual":
        factor = 1 + magnitude * np.clip(days / ramp_days, 0, 1)
    elif kind == "recurring":
        on = (days >= 0) & (np.floor(days / period_days) % 2 == 0)
        factor = np.where(on, 1 + magnitude, 1.0)
    else:
        raise ValueError(f"unknown drift kind: {kind}")
    out[TARGET] = out[TARGET] * factor
    return out


def make_drifted_prod(kind, start, **kwargs):
    """Simulated-production data with drift injected and all features rebuilt."""
    hourly = pd.read_csv("data/hourly_clean.csv", index_col=0, parse_dates=True)
    drifted = inject_drift(hourly, kind, start, **kwargs)
    return split_data(build_feature_table(drifted))[2]


if __name__ == "__main__":
    prod = pd.read_csv("data/prod.csv", index_col=0, parse_dates=True)
    start = "2010-05-01"
    d = make_drifted_prod("sudden", start)

    print("same rows as original prod:", d.index.equals(prod.index), f"({len(prod)} rows)")
    before = prod.index < start
    print("identical before start:   ", np.allclose(d.loc[before], prod.loc[before]))

    late = prod.index >= pd.Timestamp(start) + pd.Timedelta(days=8)
    for col in (TARGET, f"{TARGET}_lag_1h", f"{TARGET}_lag_168h", f"{TARGET}_roll_mean_24h"):
        print(f"{col:<38} after/before ratio: {(d.loc[late, col] / prod.loc[late, col]).mean():.3f}")