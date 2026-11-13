import pandas as pd

TARGET = "Global_active_power"
frames = [pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True) for n in ("train", "val", "prod")]
df = pd.concat(frames)

for month in (1, 8):
    sub = df[df.index.month == month][TARGET]
    print(f"\nMonth {month}")
    print(sub.groupby(sub.index.year).agg(["count", "mean", "median", "min", "max"]).round(3))