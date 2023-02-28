"""Wrappers for regression metrics from other libraries."""
import numpy as np
import scipy as sp
import statsmodels as sm
from sklearn.metrics import(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error
)


def mae(y, y_):
    return mean_absolute_error(y, y_)


def mape(y, y_):
    return mean_absolute_percentage_error(y, y_)


def mse(y, y_):
    return mean_squared_error(y, y_)


def rmse(y, y_):
    return mean_squared_error(y, y_, squared=False)


def msle(y, y_):
    return mean_squared_log_error(y, y_)


def rmsle(y, y_):
    return mean_squared_log_error(y, y_, squared=False)


def pearsonr(y, y_):
    return sp.stats.pearsonr(y, y_)


def score_report(y, y_):
    diff_y = sm.tsa.statespace.tools.diff(y)
    diff_y_ = sm.tsa.statespace.tools.diff(y_)
    scores = {
        'mae': mae(y, y_),
        'mape': mape(y, y_),
        'mse': mse(y, y_),
        'rmse': rmse(y, y_),
        'pearson': pearsonr(y, y_)[0],
        'msle': msle(y, y_),
        'rmsle': rmsle(y, y_),
        'diff_mae': mae(diff_y, diff_y_),
        'diff_mape': mape(diff_y, diff_y_),
        'diff_mse': mse(diff_y, diff_y_),
        'diff_rmse': rmse(diff_y, diff_y_),
        'diff_pearsonr': pearsonr(y, y_)[0]
    }
    return scores
