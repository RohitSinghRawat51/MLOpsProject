import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from drift import calculate_psi

TARGET = "Global_active_power"
df = pd.concat(
    pd.read_csv(f"data/{n}.csv", index_col=0, parse_dates=True) for n in ("train", "val", "prod")
)
aug = df[df.index.month == 8][TARGET]
aug_2010 = aug[aug.index.year == 2010].values

print(f"{'reference':<18}{'PSI':>8}{'KS stat':>10}{'Wasserstein':>13}")
for year in (2007, 2008, 2009):
    ref = aug[aug.index.year == year].values
    print(f"Aug {year:<14}{calculate_psi(ref, aug_2010):>8.3f}"
          f"{ks_2samp(ref, aug_2010).statistic:>10.3f}{wasserstein_distance(ref, aug_2010):>13.3f}")

for name, years in (("2007+2009", [2007, 2009]), ("2007+2008+2009", [2007, 2008, 2009])):
    ref = aug[aug.index.year.isin(years)].values
    print(f"Aug {name:<14}{calculate_psi(ref, aug_2010):>8.3f}"
          f"{ks_2samp(ref, aug_2010).statistic:>10.3f}{wasserstein_distance(ref, aug_2010):>13.3f}")