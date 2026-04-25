""" fit autoregressive models with level-dependent SGED innovations """
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from ar_ged_model import ar_design
from ar_sged_model import sged_logpdf, numerical_hessian, cov_from_hessian
from ar_sged_model import implied_sged_moments

def _std_score(x):
    """Standardize a vector to mean 0 and variance 1, guarding zero variance."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if not np.isfinite(sd) or sd <= 0:
        return x * 0.0, mu, 1.0
    return (x - mu) / sd, mu, sd

def fit_ar_sged_level(ts, driver, nar=1, trend="c"):
    """
    Fit an AR model with SGED innovations whose log(scale), log(beta), and
    log(xi) vary linearly with a standardized previous-level driver.
    """
    if nar < 1:
        raise ValueError("nar must be positive")
    ts = np.asarray(ts, dtype=float)
    driver = np.asarray(driver, dtype=float)
    if len(ts) != len(driver):
        raise ValueError("ts and driver must have the same length")
    y, x, prev_level = ar_design(ts, nar, trend)
    driver_prev = driver[nar - 1:len(ts) - 1]
    if len(driver_prev) != len(y):
        raise ValueError("driver alignment failed")
    zdriver, driver_mean, driver_sd = _std_score(driver_prev)
    if len(y) <= x.shape[1]:
        raise ValueError("not enough observations to estimate model")
    if x.shape[1] > 0:
        coeff0 = np.linalg.lstsq(x, y, rcond=None)[0]
        resid0 = y - x @ coeff0
    else:
        coeff0 = np.empty(0)
        resid0 = y.copy()
    scale0 = np.std(resid0)
    if not np.isfinite(scale0) or scale0 <= 0:
        scale0 = 1.0
    theta0 = np.concatenate([coeff0, [np.log(scale0), 0.0, np.log(1.1), 0.0,
        0.0, 0.0]])

    def unpack(theta):
        ncoef = x.shape[1]
        coeff = theta[:ncoef]
        log_scale0, log_scale1 = theta[ncoef:ncoef + 2]
        log_beta0, log_beta1 = theta[ncoef + 2:ncoef + 4]
        log_xi0, log_xi1 = theta[ncoef + 4:ncoef + 6]
        scale_t = np.exp(log_scale0 + log_scale1 * zdriver)
        beta_t = np.exp(log_beta0 + log_beta1 * zdriver)
        xi_t = np.exp(log_xi0 + log_xi1 * zdriver)
        return coeff, scale_t, beta_t, xi_t

    def neg_ll(theta):
        coeff, scale_t, beta_t, xi_t = unpack(theta)
        if (not np.all(np.isfinite(scale_t)) or not np.all(np.isfinite(beta_t)) or
                not np.all(np.isfinite(xi_t))):
            return np.inf
        resid = y - x @ coeff if x.shape[1] > 0 else y
        ll = sged_logpdf(resid, beta=beta_t, xi=xi_t, scale=scale_t).sum()
        return -ll if np.isfinite(ll) else np.inf

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)

    ncoef = x.shape[1]
    coeff, scale_t, beta_t, xi_t = unpack(res.x)
    resid = y - x @ coeff if ncoef > 0 else y
    llf = -res.fun
    npar = len(res.x)
    nfit = len(y)
    aic = 2 * npar - 2 * llf
    bic = np.log(nfit) * npar - 2 * llf
    hess_theta = numerical_hessian(neg_ll, res.x)
    cov_theta = cov_from_hessian(hess_theta)
    if cov_theta is not None and cov_theta.shape[0] == len(res.x):
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    else:
        se_theta = np.full(len(res.x), np.nan)
    return {"nar": nar, "trend": trend, "params": coeff, "resid": resid,
        "llf": llf, "aic": aic, "bic": bic, "nobs_fit": nfit,
        "prev_level": prev_level, "driver_prev": driver_prev,
        "driver_prev_std": zdriver, "driver_mean": driver_mean,
        "driver_sd": driver_sd, "theta": res.x, "se_theta": se_theta,
        "hess_theta": hess_theta, "cov_theta": cov_theta, "scale_t": scale_t,
        "beta_t": beta_t, "xi_t": xi_t}

def best_ar_sged_level(ts, driver, min_ar_order=1, max_ar_order=5, trend="c"):
    """Select the best level-dependent SGED AR fit by AIC and BIC."""
    best_aic = None
    best_bic = None
    for nar in range(min_ar_order, max_ar_order + 1):
        try:
            fit = fit_ar_sged_level(ts, driver, nar=nar, trend=trend)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if best_aic is None or fit["aic"] < best_aic["aic"]:
            best_aic = fit
        if best_bic is None or fit["bic"] < best_bic["bic"]:
            best_bic = fit
    return {"aic": best_aic, "bic": best_bic}

def print_fit_summary(label, fit):
    """Print a short summary of a selected level-dependent SGED AR fit."""
    if fit is None:
        return
    print(f"\nbest {label} lag:", fit["nar"], f"{label}:", fit[label.lower()])
    ncoef = len(fit["params"])
    se = fit["se_theta"]
    print("AR coeff:", fit["params"])
    print("\nparameter functions of standardized previous-level driver z")
    print("log(scale_t) = a0 + a1*z")
    print("log(beta_t)  = b0 + b1*z")
    print("log(xi_t)    = c0 + c1*z")
    print(f"{'param':>12}{'estimate':>12}{'stderr':>12}")
    for i, coeff in enumerate(fit["params"]):
        print(f"{('ar' + str(i)):>12}{coeff:12.3f}{se[i]:12.3f}")
    labels = ["a0", "a1", "b0", "b1", "c0", "c1"]
    vals = fit["theta"][ncoef:]
    ses = se[ncoef:]
    for name, val, err in zip(labels, vals, ses):
        print(f"{name:>12}{val:12.3f}{err:12.3f}")
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q = np.quantile(fit["driver_prev_std"], probs)
    level_source = fit["prev_level_original"] if "prev_level_original" in fit else fit["prev_level"]
    level_q = np.quantile(level_source, probs)
    a0, a1, b0, b1, c0, c1 = vals
    scale_q = np.exp(a0 + a1 * q)
    beta_q = np.exp(b0 + b1 * q)
    xi_q = np.exp(c0 + c1 * q)
    print("\nparameter values across driver quantiles")
    print(f"{'q':>8}{'level':>12}{'scale':>12}{'beta':>12}{'xi':>12}"
        f"{'skew':>12}{'ex kurt':>12}")
    qlabels = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
    for qlab, lev, sc, be, xi in zip(qlabels, level_q, scale_q, beta_q, xi_q):
        moments = implied_sged_moments(be, xi, sc)
        print(f"{qlab:>8}{lev:12.3f}{sc:12.3f}{be:12.3f}{xi:12.3f}"
            f"{moments['skew']:12.3f}{moments['excess_kurtosis']:12.3f}")
