""" statistics utilities """
import numpy as np
import numpy.typing as npt
from util import print_vec
import statsmodels.api as sm

def print_acf(x: npt.NDArray, nlags: int = 5, title="ACF\n",
    with_lags=True, trailer=None, end=None, fmt_acf="%6.3f", fmt_lag="%6d"):
    """ print the autocorrelations of time series x """
    if nlags < 1:
        return
    print_vec(acf(x, nlags=nlags), with_num=with_lags, num0=1, title=title,
        fmt=fmt_acf, fmt_num=fmt_lag, trailer=trailer)
    if end:
        print(end=end)

def acf(x: npt.NDArray, nlags: int = 5):
    """ return autocorrelations of time series x """
    return sm.tsa.acf(x, nlags=nlags)[1:]

