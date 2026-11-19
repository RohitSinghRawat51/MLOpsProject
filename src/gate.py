import numpy as np
from scipy.stats import norm


def diebold_mariano(err_champion, err_challenger, lags=24):
    """Diebold-Mariano test on absolute-error loss.

    d_t = champion loss - challenger loss. A positive statistic means the
    challenger is better. Hourly errors are autocorrelated, so the variance
    uses a Bartlett-weighted (HAC) estimate over `lags` lags, not the plain one.
    """
    d = np.asarray(err_champion, dtype=float) - np.asarray(err_challenger, dtype=float)
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:                              # identical models: nothing to test
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def should_promote(err_champion, err_challenger, min_gain=0.05, alpha=0.05):
    """Promote only if the challenger is significantly AND meaningfully better."""
    stat, p = diebold_mariano(err_champion, err_challenger)
    gain = 1 - float(np.mean(err_challenger)) / float(np.mean(err_champion))
    return {
        "promote": bool(stat > 0 and p < alpha and gain >= min_gain),
        "dm_stat": stat, "p_value": p, "gain": gain,
    }