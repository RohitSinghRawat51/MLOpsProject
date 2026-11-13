import itertools
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from drift import calculate_psi

TARGET = "Global_active_power"
df = pd.concat(
    pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True) for n in ("train", "val", "prod")
)


def month_values(year, month):
    sub = df[(df.index.year == year) & (df.index.month == month)][TARGET]
    return sub.values


rows = []
for month in range(1, 12):  # 2010 data ends in late November
    pairs = [
        (month_values(a, month), month_values(b, month))
        for a, b in itertools.combinations((2007, 2008, 2009), 2)
    ]
    cur, prev = month_values(2010, month), month_values(2009, month)
    rows.append({
        "month": month,
        "natural_max_PSI": max(calculate_psi(x, y) for x, y in pairs),
        "2010v2009_PSI": calculate_psi(prev, cur),
        "natural_max_KS": max(ks_2samp(x, y).statistic for x, y in pairs),
        "2010v2009_KS": ks_2samp(prev, cur).statistic,
        "natural_max_W": max(wasserstein_distance(x, y) for x, y in pairs),
        "2010v2009_W": wasserstein_distance(prev, cur),
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))