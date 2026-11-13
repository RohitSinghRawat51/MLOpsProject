import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TARGET = "Global_active_power"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "Global_active_power_lag_1h", "Global_active_power_lag_24h", "Global_active_power_lag_168h",
    "Global_active_power_roll_mean_24h", "Global_active_power_roll_std_24h",
]
read = lambda n: pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True)
train, val, prod = read("train"), read("val"), read("prod")

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(train[FEATURES], train[TARGET])


def abs_errors(df):
    return (df[TARGET] - model.predict(df[FEATURES])).abs()


e09, e10 = abs_errors(val), abs_errors(prod)
print(f"Overall 2009 MAE (should be 0.3590): {e09.mean():.4f}")

out = pd.DataFrame({
    "MAE_2009": e09.groupby(e09.index.month).mean(),
    "MAE_2010": e10.groupby(e10.index.month).mean(),
})
out["change"] = out["MAE_2010"] - out["MAE_2009"]
print(out.round(3))