import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):
  
    # Build bin edges from the reference data's own quantiles, so each
    # reference bin starts with roughly equal counts of reference data
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # avoid divide-by-zero / log(0) for empty bins
    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

def month_values(df, target, month):
    """All values from a given calendar month, across every year in df."""
    return df[df.index.month == month][target].values

def psi_table(reference, current, bins=10):
    """Same calculation as calculate_psi, but returns the per-bin breakdown."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-4
    ref_adj = np.where(ref_pct == 0, eps, ref_pct)
    cur_adj = np.where(cur_pct == 0, eps, cur_pct)
    contrib = (cur_adj - ref_adj) * np.log(cur_adj / ref_adj)
    return pd.DataFrame({
        "bin_upper_edge": edges[1:],
        "ref_%": (ref_pct * 100).round(1),
        "cur_%": (cur_pct * 100).round(1),
        "psi_contribution": contrib.round(3),
    })

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
    prod = pd.read_csv("data/prod.csv", index_col=0, parse_dates=True)

    TARGET = "Global_active_power"
    reference = train[TARGET].values

    # Sanity check 1: compare the reference to itself -- should be ~0
    self_psi = calculate_psi(reference, reference)
    print(f"PSI (train vs itself, sanity check): {self_psi:.4f}")

    # Sanity check 2: compare train to the first month of simulated production
    first_month = prod.loc["2010-01-01":"2010-01-31", TARGET].values
    psi_jan = calculate_psi(reference, first_month)
    print(f"PSI (train vs Jan 2010):             {psi_jan:.4f}")

    # Sanity check 3: compare train to a later month, for contrast
    august = prod.loc["2010-08-01":"2010-08-31", TARGET].values
    psi_aug = calculate_psi(reference, august)
    print(f"PSI (train vs Aug 2010):              {psi_aug:.4f}")

    # Same-season comparison: train's own Januaries vs 2010's January,
    # and train's own Augusts vs 2010's August -- to test whether the
    # earlier huge PSI values were really just seasonality, not drift.
    jan_reference = month_values(train, TARGET, 1)
    psi_jan_same_season = calculate_psi(jan_reference, first_month)
    print(f"\nPSI (train Januaries only vs Jan 2010): {psi_jan_same_season:.4f}")

    aug_reference = month_values(train, TARGET, 8)
    psi_aug_same_season = calculate_psi(aug_reference, august)
    print(f"PSI (train Augusts only vs Aug 2010):   {psi_aug_same_season:.4f}")

    print("\nAugust breakdown (train Augusts vs Aug 2010):")
    print(psi_table(aug_reference, august).to_string(index=False))
    print("\nJanuary breakdown (train Januaries vs Jan 2010):")
    print(psi_table(jan_reference, first_month).to_string(index=False))