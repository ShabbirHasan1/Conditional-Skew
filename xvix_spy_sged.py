""" analyze VIX and SPY with VIX-driven SGED models """
import time
import numpy as np
import pandas as pd
from datetime import date
from scipy import stats
from scipy.optimize import minimize
from pandas_util import read_csv_date_index, print_first_last
from stats import print_acf
from ar_noise_report import print_noise_stats_by_prev_level_quantile
from ar_sged_model import (best_ar_sged, print_fit_summary as print_const_summary,
    sged_logpdf, numerical_hessian, cov_from_hessian, implied_sged_moments)
from ar_sged_level_model import (best_ar_sged_level,
    print_fit_summary as print_level_summary)

start = time.time()
pd.set_option("display.float_format", "{:.3f}".format)
np.set_printoptions(precision=3)

infile = "vix_spy.csv"
min_ar_order = 1
max_ar_order = 1
trend = "c"
nacf = 12
n_vix_bins = 2
date_min = date(1900, 1, 1)
date_max = date(2100, 12, 31)
driver_kind = "log_level"
show_aic = False
show_bic = True
show_spy_returns = True
show_spy_returns_over_vix = True

print("data file:", infile)
print("max AR order:", max_ar_order)
print("trend:", trend)
print("driver kind:", driver_kind)
print("VIX bins:", n_vix_bins)
print("date_min:", date_min)
print("date_max:", date_max)
print("show_spy_returns:", show_spy_returns)
print("show_spy_returns_over_vix:", show_spy_returns_over_vix)

def print_series_stats(label, x):
    x = np.asarray(x, dtype=float)
    print(label)
    print(f"{'#obs':>10}{'mean':>10}{'sd':>10}{'skew':>10}{'ex kurt':>10}")
    print(f"{len(x):10d}{np.mean(x):10.4f}{np.std(x):10.4f}{stats.skew(x):10.3f}"
        f"{stats.kurtosis(x):10.3f}")

def fit_sged_constant(x):
    x = np.asarray(x, dtype=float)
    mu0 = np.mean(x)
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0:
        scale0 = 1.0
    theta0 = np.array([mu0, np.log(scale0), np.log(1.1), 0.0], dtype=float)

    def neg_ll(theta):
        mu = theta[0]
        scale = np.exp(theta[1])
        beta = np.exp(theta[2])
        xi = np.exp(theta[3])
        if not np.isfinite(scale) or not np.isfinite(beta) or not np.isfinite(xi):
            return np.inf
        ll = sged_logpdf(x - mu, beta=beta, xi=xi, scale=scale).sum()
        return -ll if np.isfinite(ll) else np.inf

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    hess = numerical_hessian(neg_ll, res.x)
    cov = cov_from_hessian(hess)
    if cov is not None:
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        se = np.full(len(res.x), np.nan)
    mu = res.x[0]
    scale = np.exp(res.x[1])
    beta = np.exp(res.x[2])
    xi = np.exp(res.x[3])
    llf = -res.fun
    npar = 4
    nobs = len(x)
    implied = implied_sged_moments(beta, xi, scale)
    return {"mu": mu, "scale": scale, "beta": beta, "xi": xi,
        "theta": res.x, "se": se, "llf": llf, "aic": 2*npar - 2*llf,
        "bic": np.log(nobs)*npar - 2*llf, "implied_skew": implied["skew"],
        "implied_excess_kurtosis": implied["excess_kurtosis"]}

def fit_sged_level(x, driver):
    x = np.asarray(x, dtype=float)
    driver = np.asarray(driver, dtype=float)
    z = (driver - np.mean(driver)) / np.std(driver)
    mu0 = np.mean(x)
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0:
        scale0 = 1.0
    theta0 = np.array([mu0, np.log(scale0), 0.0, np.log(1.1), 0.0, 0.0, 0.0],
        dtype=float)

    def unpack(theta):
        mu = theta[0]
        a0, a1, b0, b1, c0, c1 = theta[1:]
        scale_t = np.exp(a0 + a1*z)
        beta_t = np.exp(b0 + b1*z)
        xi_t = np.exp(c0 + c1*z)
        return mu, scale_t, beta_t, xi_t

    def neg_ll(theta):
        mu, scale_t, beta_t, xi_t = unpack(theta)
        if (not np.all(np.isfinite(scale_t)) or not np.all(np.isfinite(beta_t)) or
                not np.all(np.isfinite(xi_t))):
            return np.inf
        ll = sged_logpdf(x - mu, beta=beta_t, xi=xi_t, scale=scale_t).sum()
        return -ll if np.isfinite(ll) else np.inf

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    hess = numerical_hessian(neg_ll, res.x)
    cov = cov_from_hessian(hess)
    if cov is not None:
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        se = np.full(len(res.x), np.nan)
    mu, scale_t, beta_t, xi_t = unpack(res.x)
    llf = -res.fun
    npar = len(res.x)
    nobs = len(x)
    return {"mu": mu, "theta": res.x, "se": se, "scale_t": scale_t,
        "beta_t": beta_t, "xi_t": xi_t, "driver": driver, "driver_std": z,
        "llf": llf, "aic": 2*npar - 2*llf, "bic": np.log(nobs)*npar - 2*llf}

def print_sged_constant_summary(label, fit):
    print(f"\n{label}")
    print("mu:", f"{fit['mu']:.6f}", "scale:", f"{fit['scale']:.6f}",
        "beta:", f"{fit['beta']:.3f}", "xi:", f"{fit['xi']:.3f}")
    print("AIC:", f"{fit['aic']:.3f}", "BIC:", f"{fit['bic']:.3f}")
    print("implied skew:", f"{fit['implied_skew']:.3f}",
        "implied excess kurtosis:", f"{fit['implied_excess_kurtosis']:.3f}")
    print(f"{'param':>10}{'estimate':>12}{'stderr':>12}")
    for name, val, err in zip(["mu", "log_sd", "log_beta", "log_xi"],
            fit["theta"], fit["se"]):
        print(f"{name:>10}{val:12.4f}{err:12.4f}")

def print_sged_level_summary(label, fit, vix_prev_original):
    print(f"\n{label}")
    print("mu:", f"{fit['mu']:.6f}")
    print("AIC:", f"{fit['aic']:.3f}", "BIC:", f"{fit['bic']:.3f}")
    print("parameter functions of standardized log(VIX_{t-1}) driver z")
    print("log(scale_t) = a0 + a1*z")
    print("log(beta_t)  = b0 + b1*z")
    print("log(xi_t)    = c0 + c1*z")
    print(f"{'param':>10}{'estimate':>12}{'stderr':>12}")
    for name, val, err in zip(["mu", "a0", "a1", "b0", "b1", "c0", "c1"],
            fit["theta"], fit["se"]):
        print(f"{name:>10}{val:12.4f}{err:12.4f}")
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qlabels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
    zq = np.quantile(fit["driver_std"], probs)
    level_q = np.quantile(vix_prev_original, probs)
    a0, a1, b0, b1, c0, c1 = fit["theta"][1:]
    scale_q = np.exp(a0 + a1*zq)
    beta_q = np.exp(b0 + b1*zq)
    xi_q = np.exp(c0 + c1*zq)
    print("\nparameter values across driver quantiles")
    print(f"{'q':>8}{'VIX':>12}{'scale':>12}{'beta':>12}{'xi':>12}"
        f"{'skew':>12}{'ex kurt':>12}")
    for qlab, lev, sc, be, xi in zip(qlabels, level_q, scale_q, beta_q, xi_q):
        moments = implied_sged_moments(be, xi, sc)
        print(f"{qlab:>8}{lev:12.3f}{sc:12.6f}{be:12.3f}{xi:12.3f}"
            f"{moments['skew']:12.3f}{moments['excess_kurtosis']:12.3f}")

def print_bic_comparison(label, fit_const, fit_level):
    delta_bic = fit_level["bic"] - fit_const["bic"]
    print(f"\nBIC comparison for {label}")
    print("constant SGED BIC:", f"{fit_const['bic']:.3f}")
    print("level-dependent SGED BIC:", f"{fit_level['bic']:.3f}")
    print("delta BIC (level - constant):", f"{delta_bic:.3f}")
    if delta_bic < 0:
        print("BIC favors level-dependent SGED.")
    elif delta_bic > 0:
        print("BIC favors constant SGED.")
    else:
        print("BIC is tied.")

df = read_csv_date_index(infile, date_min=date_min, date_max=date_max)
df = df.apply(pd.to_numeric, errors="coerce").astype(float)
print_first_last(df)
print(df.describe())

col_vix = df.columns[0]
col_spy = df.columns[1]
vix = df[col_vix]
spy = df[col_spy]

# First column: autoregression as before on log(VIX)
vix_log = np.log(vix.where(vix > 0))
vix_log = vix_log.dropna()
vix_original = vix.loc[vix_log.index]
driver_vix = vix_log.to_numpy() if driver_kind == "log_level" else vix_original.to_numpy()
print("\nsymbol:", col_vix)
print("#obs:", len(vix_log))
print_acf(vix_log.to_numpy(), nlags=nacf, title="\nACF\n")
fits_const_vix = best_ar_sged(vix_log.to_numpy(), min_ar_order=min_ar_order,
    max_ar_order=max_ar_order, trend=trend)
fits_level_vix = best_ar_sged_level(vix_log.to_numpy(), driver_vix,
    min_ar_order=min_ar_order, max_ar_order=max_ar_order, trend=trend)
if show_bic:
    print("\nconstant-parameter SGED benchmark")
    print_const_summary("BIC", fits_const_vix["bic"])
    fit_vix_level = fits_level_vix["bic"]
    if fit_vix_level is not None:
        fit_vix_level["prev_level_original"] = vix_original.to_numpy()[fit_vix_level["nar"] - 1:-1]
    print_level_summary("BIC", fit_vix_level)
    if fits_const_vix["bic"] is not None and fits_level_vix["bic"] is not None:
        delta_bic = fits_level_vix["bic"]["bic"] - fits_const_vix["bic"]["bic"]
        print("\nBIC comparison")
        print("constant-parameter SGED BIC:", f"{fits_const_vix['bic']['bic']:.3f}")
        print("level-dependent SGED BIC:", f"{fits_level_vix['bic']['bic']:.3f}")
        print("delta BIC (level - constant):", f"{delta_bic:.3f}")
        if delta_bic < 0:
            print("BIC favors level-dependent noise parameters.")
        else:
            print("BIC does not favor level-dependent noise parameters.")

# Second column: returns and returns/VIX(t-1)
spy_ret = spy.pct_change()
df_ret = pd.DataFrame({"ret": spy_ret, "vix_prev": vix.shift(1),
    "log_vix_prev": np.log(vix.shift(1))})
df_ret = df_ret.dropna()
df_ret["ret_over_vix"] = df_ret["ret"] / df_ret["vix_prev"]

series_to_run = []
if show_spy_returns:
    series_to_run.append("ret")
if show_spy_returns_over_vix:
    series_to_run.append("ret_over_vix")

for series_name in series_to_run:
    x = df_ret[series_name].to_numpy()
    vix_prev = df_ret["vix_prev"].to_numpy()
    log_vix_prev = df_ret["log_vix_prev"].to_numpy()
    label = "SPY returns" if series_name == "ret" else "SPY returns / VIX(t-1)"
    print("\n" + label)
    print_series_stats("overall", x)
    print_noise_stats_by_prev_level_quantile(label, log_vix_prev, x, n_vix_bins,
        vix_prev)
    fit_const = fit_sged_constant(x)
    fit_level = fit_sged_level(x, log_vix_prev)
    print_sged_constant_summary("global SGED fit", fit_const)
    print_sged_level_summary("SGED with parameters depending on log(VIX)", fit_level,
        vix_prev)
    print_bic_comparison(label, fit_const, fit_level)

print("\ntime elapsed (sec):", f"{time.time() - start:0.2f}")
