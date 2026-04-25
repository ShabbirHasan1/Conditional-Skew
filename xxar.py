""" fit an AR model to each time series in a csv file """
import numpy as np
import pandas as pd
from pandas_util import read_csv_date_index, print_first_last
from stats import print_acf
from statsmodels_util import best_ar, print_best_ar
from ar_noise_report import (print_noise_stats,
    print_noise_stats_by_prev_level_quantile, prev_level_aligned)

pd.set_option("display.float_format", "{:.3f}".format)
np.set_printoptions(precision=3)

infile = "vix.csv"
min_ar_order = 1
max_ar_order = 1
trend = "c"
take_logs = True
nacf = 12
print_model_summaries = False
show_aic = False
show_bic = True
n_prev_level_quantiles = 2
print("data file:", infile)
print("max AR order:", max_ar_order)
print("previous-level quantiles:", n_prev_level_quantiles)

df = read_csv_date_index(infile)
df = df.apply(pd.to_numeric, errors="coerce").astype(float)
df_original = df.copy()
if take_logs:
    df = np.log(df.where(df > 0))

print_first_last(df)
print(df.describe())

for col in df.columns:
    ts = df[col].dropna().to_numpy()
    levels_original = df_original[col].dropna().to_numpy()
    print("\nsymbol:", col)
    print("#obs:", len(ts))
    if len(ts) <= max_ar_order:
        print("not enough observations to fit AR(", max_ar_order, ")", sep="")
        continue
    print_acf(ts, nlags=nacf, title="\nACF\n")
    print_best_ar(ts, min_ar_order=min_ar_order, max_ar_order=max_ar_order,
        trend=trend, print_best_summary=print_model_summaries,
        print_summary=False, aic=show_aic, bic=show_bic)
    fits = best_ar(ts, min_ar_order=min_ar_order, max_ar_order=max_ar_order,
        trend=trend, print_summary=False)
    if show_aic:
        model_fit = fits["aic"][2]
        if model_fit is not None:
            resid = np.asarray(model_fit.resid)
            prev_level, prev_level_original = prev_level_aligned(ts, resid,
                levels_original)
            print_noise_stats("\nAIC noise", resid)
            print_noise_stats_by_prev_level_quantile("\nAIC noise", prev_level,
                resid, n_prev_level_quantiles, prev_level_original)
    if show_bic:
        model_fit = fits["bic"][2]
        if model_fit is not None:
            resid = np.asarray(model_fit.resid)
            prev_level, prev_level_original = prev_level_aligned(ts, resid,
                levels_original)
            print_noise_stats("\nBIC noise", resid)
            print_noise_stats_by_prev_level_quantile("\nBIC noise", prev_level,
                resid, n_prev_level_quantiles, prev_level_original)
