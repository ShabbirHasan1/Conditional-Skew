""" fit AR models with level-dependent SGED innovations """
import time
import numpy as np
import pandas as pd
from datetime import date
from pandas_util import read_csv_date_index, print_first_last
from stats import print_acf
from ar_sged_model import best_ar_sged, print_fit_summary as print_const_summary
from ar_sged_level_model import (best_ar_sged_level,
    print_fit_summary as print_level_summary)

start = time.time()
pd.set_option("display.float_format", "{:.3f}".format)
np.set_printoptions(precision=3)

infile = "vix.csv"
min_ar_order = 1
max_ar_order = 1
trend = "c"
take_logs = True
nacf = 12
show_constant_benchmark = True
show_aic = False
show_bic = True
driver_kind = "log_level"  # "log_level" or "level"
date_min = date(1900, 1, 1)
date_max = date(2100, 12, 31)

print("data file:", infile)
print("max AR order:", max_ar_order)
print("trend:", trend)
print("driver kind:", driver_kind)
print("date_min:", date_min)
print("date_max:", date_max)

df = read_csv_date_index(infile, date_min=date_min, date_max=date_max)
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
    if driver_kind == "log_level":
        if take_logs:
            driver = ts.copy()
        else:
            driver = np.log(np.maximum(levels_original, 1.0e-12))
    elif driver_kind == "level":
        driver = levels_original.copy()
    else:
        raise ValueError("driver_kind must be 'log_level' or 'level'")
    print("\nsymbol:", col)
    print("#obs:", len(ts))
    if len(ts) <= max_ar_order + 8:
        print("not enough observations to fit AR(", max_ar_order,
            ") level-dependent SGED", sep="")
        continue
    print_acf(ts, nlags=nacf, title="\nACF\n")
    if show_constant_benchmark:
        fits_const = best_ar_sged(ts, min_ar_order=min_ar_order,
            max_ar_order=max_ar_order, trend=trend)
        if show_aic:
            print("\nconstant-parameter SGED benchmark")
            print_const_summary("AIC", fits_const["aic"])
        if show_bic:
            print("\nconstant-parameter SGED benchmark")
            print_const_summary("BIC", fits_const["bic"])
    fits_level = best_ar_sged_level(ts, driver, min_ar_order=min_ar_order,
        max_ar_order=max_ar_order, trend=trend)
    if show_aic:
        fit = fits_level["aic"]
        if fit is not None:
            fit["prev_level_original"] = levels_original[fit["nar"] - 1:-1]
        print_level_summary("AIC", fit)
    if show_bic:
        fit = fits_level["bic"]
        if fit is not None:
            fit["prev_level_original"] = levels_original[fit["nar"] - 1:-1]
        print_level_summary("BIC", fit)
    if show_bic and show_constant_benchmark:
        fit_const = fits_const["bic"]
        fit_level = fits_level["bic"]
        if fit_const is not None and fit_level is not None:
            delta_bic = fit_level["bic"] - fit_const["bic"]
            print("\nBIC comparison")
            print("constant-parameter SGED BIC:", f"{fit_const['bic']:.3f}")
            print("level-dependent SGED BIC:", f"{fit_level['bic']:.3f}")
            print("delta BIC (level - constant):", f"{delta_bic:.3f}")
            if delta_bic < 0:
                print("BIC favors level-dependent noise parameters.")
            elif delta_bic > 0:
                print("BIC favors constant noise parameters.")
            else:
                print("BIC is tied.")

print("\ntime elapsed (sec):", f"{time.time() - start:0.2f}")
