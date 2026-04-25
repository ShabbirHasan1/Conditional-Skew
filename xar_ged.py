""" fit AR models with GED innovations to each time series in a csv file """
import numpy as np
import pandas as pd
from pandas_util import read_csv_date_index, print_first_last
from stats import print_acf
from ar_ged_model import best_ar_ged, print_fit_summary
from ar_noise_report import (print_noise_stats,
    print_noise_stats_by_prev_level_quantile)

pd.set_option("display.float_format", "{:.3f}".format)
np.set_printoptions(precision=3)

infile = "vix.csv"
min_ar_order = 1
max_ar_order = 1
trend = "c"
take_logs = True
nacf = 12
show_aic = False
show_bic = True
n_prev_level_quantiles = 2

print("data file:", infile)
print("max AR order:", max_ar_order)
print("trend:", trend)
print("previous-level quantiles:", n_prev_level_quantiles)

df = read_csv_date_index(infile)
df = df.apply(pd.to_numeric, errors="coerce").astype(float)
df_original = df.copy()
if take_logs:
    df = np.log(df.where(df > 0))

print_first_last(df)
print(df.describe())

for col in df.columns:
    ts_ser = df[col].dropna()
    ts = ts_ser.to_numpy()
    levels_original = df_original.loc[ts_ser.index, col].to_numpy()
    print("\nsymbol:", col)
    print("#obs:", len(ts))
    if len(ts) <= max_ar_order + 2:
        print("not enough observations to fit AR(", max_ar_order, ") GED", sep="")
        continue
    print_acf(ts, nlags=nacf, title="\nACF\n")
    fits = best_ar_ged(ts, min_ar_order=min_ar_order, max_ar_order=max_ar_order,
        trend=trend)
    if show_aic:
        fit = fits["aic"]
        print_fit_summary("AIC", fit)
        if fit is not None:
            print_noise_stats("\nAIC noise", fit["resid"])
            prev_level_original = levels_original[fit["nar"] - 1:-1]
            print_noise_stats_by_prev_level_quantile("\nAIC noise",
                fit["prev_level"], fit["resid"], n_prev_level_quantiles,
                prev_level_original)
    if show_bic:
        fit = fits["bic"]
        print_fit_summary("BIC", fit)
        if fit is not None:
            print_noise_stats("\nBIC noise", fit["resid"])
            prev_level_original = levels_original[fit["nar"] - 1:-1]
            print_noise_stats_by_prev_level_quantile("\nBIC noise",
                fit["prev_level"], fit["resid"], n_prev_level_quantiles,
                prev_level_original)
