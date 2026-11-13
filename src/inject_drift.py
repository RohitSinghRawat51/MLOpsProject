import numpy as np
import pandas as pd
from features import add_lag_features, add_rolling_features
TARGET = "Global_active_power"


def inject_drift(df, kind, start, magnitude=0.5, ramp_days=60, period_days=60):
    """Return a copy of df with synthetic drift applied to TARGET from `start`.

    The lag and rolling features are rebuilt from the drifted target, so every
    feature stays consistent with it.

    kind:
      sudden      multiply consumption by (1 + magnitude) from `start` on
      gradual     same, but the multiplier ramps in linearly over ramp_days
      recurring   the sudden shift is on for period_days, off for period_days, repeating
    """
    out = df.copy()
    start = pd.Timestamp(start)
    days = (out.index - start).total_seconds() / 86400
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

    base = out[[TARGET]].copy()
    base = add_lag_features(base)
    base = add_rolling_features(base)
    for col in base.columns:
        if col != TARGET:
            out[col] = base[col].combine_first(out[col])
    return out




if __name__ == "__main__":
    prod = pd.read_csv("data/prod.csv", index_col=0, parse_dates=True)
    for kind in ("sudden", "gradual", "recurring"):
        d = inject_drift(prod, kind, "2010-05-01")
        before = d.loc[:"2010-04-30", TARGET].mean()
        after = d.loc["2010-08-01":, TARGET].mean()
        print(f"{kind:<10} mean before start: {before:.3f}   mean from Aug: {after:.3f}")