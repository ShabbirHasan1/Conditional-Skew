""" statsmodel utilities """

from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm

def best_ar(data, min_ar_order, max_ar_order=10, trend="n", print_params=False,
    print_summary=True):
    """
    Function to find the best AR (autoregressive) model order based on AIC and BIC.

    Parameters:
    data (array-like): The time series data for which the AR model is to be fitted.
    min_ar_order (int, optional): The minimum AR model order to be considered. The default is 1.
    max_ar_order (int, optional): The maximum AR model order to be considered. The default is 10.
    print_params (bool, optional): Whether to print the parameters of each AR model estimated. The default is False.
    trend (string, optional): "n" - No trend; "c" - Constant only; "t" - Time trend only;
        "ct" - Constant and time trend.

    Returns:
    dict: A dictionary with the best model order, AIC/BIC values, and model instance for both AIC and BIC criteria.
    """
    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = 0
    best_order_bic = 0
    best_model_aic = None
    best_model_bic = None
    if print_params:
        print(end="\n")
    for order in range(min_ar_order, max_ar_order + 1):
        try:
            model = sm.tsa.AutoReg(data, lags=order, trend=trend)
            model_fit = model.fit()
        except ValueError:
            continue
        aic = model_fit.aic
        bic = model_fit.bic
        if print_params:
            print(f'Order {order}, AIC: {aic}, BIC: {bic}')
            print('Coefficients:', model_fit.params, "\n")
        if print_summary:
            print(model_fit.summary(), end="\n\n")
        if aic < best_aic:
            best_aic = aic
            best_order_aic = order
            best_model_aic = model_fit
        if bic < best_bic:
            best_bic = bic
            best_order_bic = order
            best_model_bic = model_fit
    return {'aic': (best_order_aic, best_aic, best_model_aic), 'bic': (best_order_bic, best_bic, best_model_bic)}

def print_best_ar(x, min_ar_order=1, max_ar_order=5, trend="n",
    print_best_params=True, print_best_summary=False, print_params=False,
    print_summary=True, aic=True, bic=True):
    """ wrapper that prints the results from best_ar """
    if max_ar_order < min_ar_order or max_ar_order < 0:
        return
    fits = best_ar(x, min_ar_order=min_ar_order, max_ar_order=max_ar_order,
        trend=trend, print_params=print_params, print_summary=print_summary)
    if aic:
        aic_fit = fits["aic"]
        aic_model_fit = aic_fit[2]
        print("\nbest AIC lag:", aic_fit[0], "AIC:", aic_fit[1])
    if bic:
        bic_fit = fits["bic"]
        bic_model_fit = bic_fit[2]
    if aic and print_best_params:
        print("AR coeff:", aic_model_fit.params)
    if aic and print_best_summary:
        print(aic_model_fit.summary())
    if bic:
        print("\nbest BIC lag:", bic_fit[0], "BIC:", bic_fit[1])
    if print_best_params:
        print("AR coeff:", bic_model_fit.params)
    if print_best_summary:
        print(bic_model_fit.summary())
