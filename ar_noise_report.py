""" helpers for printing residual diagnostics """
import numpy as np
import pandas as pd
from scipy import stats

def permutation_pvalue_stat_diff(x1, x2, stat_func, nperm=2000, seed=0):
    """Two-sided permutation p-value for a difference in a sample statistic."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    obs = abs(stat_func(x1) - stat_func(x2))
    pooled = np.concatenate([x1, x2])
    n1 = len(x1)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(nperm):
        perm = rng.permutation(pooled)
        diff = abs(stat_func(perm[:n1]) - stat_func(perm[n1:]))
        if diff >= obs:
            count += 1
    return (count + 1.0) / (nperm + 1.0)

def print_noise_stats(label, resid):
    """Print standard deviation, skew, and excess kurtosis for residuals."""
    print(label)
    print(f"{'sd':>10}{'skew':>10}{'ex kurt':>10}")
    print(f"{np.std(resid):10.3f}{stats.skew(resid):10.3f}{stats.kurtosis(resid):10.3f}")

def prev_level_aligned(ts, resid, levels_original=None):
    """
    Align residuals with the immediately previous observed level.
    Returns previous levels in model units and, optionally, original units.
    """
    nresid = len(resid)
    if nresid < 1:
        return np.array([]), np.array([])
    prev_level = ts[-nresid-1:-1]
    if levels_original is None:
        prev_level_original = prev_level
    else:
        prev_level_original = levels_original[-nresid-1:-1]
    return prev_level, prev_level_original

def print_noise_stats_by_prev_level_quantile(label, prev_level, resid, n_quantiles,
    prev_level_original):
    """Print residual diagnostics by quantiles of previous observed level."""
    nresid = len(resid)
    if nresid < 2:
        return
    df_noise = pd.DataFrame({"prev_level": prev_level, "prev_level_original":
        prev_level_original, "resid": resid})
    try:
        df_noise["quantile"] = pd.qcut(df_noise["prev_level"], q=n_quantiles,
            labels=False, duplicates="drop")
    except ValueError:
        return
    print(label + " by previous level quantile")
    print(f"{'':>12}{'level':>30}{'noise':>30}")
    print(f"{'q':>4}{'#obs':>8}{'min':>10}{'mean':>10}{'max':>10}"
        f"{'mean':>10}{'sd':>10}{'skew':>10}{'ex kurt':>10}")
    grouped = list(df_noise.groupby("quantile", observed=False))
    noise_groups = []
    for iquant, df_group in grouped:
        x = df_group["resid"].to_numpy()
        noise_groups.append(x)
        xlevel = df_group["prev_level_original"].to_numpy()
        print(f"{int(iquant) + 1:4d}{len(x):8d}{np.min(xlevel):10.3f}"
            f"{np.mean(xlevel):10.3f}{np.max(xlevel):10.3f}{np.mean(x):10.3f}"
            f"{np.std(x):10.3f}{stats.skew(x):10.3f}{stats.kurtosis(x):10.3f}")
    if len(noise_groups) == 2:
        x1, x2 = noise_groups
        print("\np-values for equal noise properties in bins 1 and 2")
        print(f"{'mean':>12}{'sd':>12}{'skew':>12}{'ex kurt':>12}")
        p_mean = permutation_pvalue_stat_diff(x1, x2, np.mean)
        p_sd = permutation_pvalue_stat_diff(x1, x2, np.std, seed=1)
        p_skew = permutation_pvalue_stat_diff(x1, x2, stats.skew, seed=2)
        p_kurt = permutation_pvalue_stat_diff(x1, x2, stats.kurtosis, seed=3)
        print(f"{p_mean:12.4g}{p_sd:12.4g}{p_skew:12.4g}{p_kurt:12.4g}")
