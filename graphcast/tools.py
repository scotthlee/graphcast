import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as stm
import itertools
import warnings

from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict

from . import metrics
from .models import statsmodels as models


def strip_apostrophes(series):
    return [str(s).replace("'", "") for s in series]


def ts_plots(trend, sd=None, sdp=1):
    '''Makes ACF and PACF plots for a time series.'''
    # Differencing
    diff1 = stm.tsa.statespace.tools.diff(trend,
                                          k_seasonal_diff=sd,
                                          seasonal_periods=sdp)
    diff2 = stm.tsa.statespace.tools.diff(diff1,
                                          k_seasonal_diff=sd,
                                          seasonal_periods=sdp)
    
    # Basic plots
    time = np.arange(0, trend.shape[0])
    fig, axes = plt.subplots(3, 3)
    
    # Making the plots
    axes[0, 0].plot(time, trend)
    axes[0, 0].set_title('Original')
    axes[0, 1].plot(time[:diff1.shape[0]], diff1)
    axes[0, 1].set_title('Diff1')
    axes[0, 2].plot(time[:diff2.shape[0]], diff2)
    axes[0, 2].set_title('Diff2')
    stm.graphics.tsa.plot_acf(trend, ax=axes[1, 0])
    stm.graphics.tsa.plot_acf(diff1, ax=axes[1, 1])
    stm.graphics.tsa.plot_acf(diff2, ax=axes[1, 2])
    stm.graphics.tsa.plot_pacf(trend, ax=axes[2, 0], method='ywm')
    stm.graphics.tsa.plot_pacf(diff1, ax=axes[2, 1], method='ywm')
    stm.graphics.tsa.plot_pacf(diff2, ax=axes[2, 2], method='ywm')
    
    plt.tight_layout()
    plt.show()


def prediction_plot(y, model, horizons, panel=False):
    keys = ['test_' + str(h) for h in horizons]
    casts = [model.forecasts[k] for k in keys]
    idx = [model.indices[k] for k in keys]
    if panel:
        fig, axes = plt.subplots(1, len(casts), sharey=True)
        for i, ax in enumerate(axes):
            ax.plot(range(len(y)), y)
            ax.scatter(idx[i], casts[i])
    else:
        plt.plot(range(len(y)), y)
        for i, c in enumerate(casts):
            plt.scatter(idx[i], c, s=10)
    plt.tight_layout()
    plt.show()
    return

def score_dict_to_df(d, model=None, split=None):
    '''Converts a dictionary of regression metrics to a data frame.'''
    df = pd.DataFrame(list(d.values())).transpose()
    df.columns = list(d.keys())
    if split is not None:
        df['split'] = split
    return df


def scores_to_df(model, drop_val=True):
    '''Converts multiple dictionaries of score metrics to data frames.'''
    d = model.scores
    splits = list(d.keys())
    metrics = list(d[splits[0]].keys())
    if drop_val:
        splits.remove('val')
    horizons = [s.split('_')[1] for s in splits]
    out = []
    for s in splits:
        out.append(score_dict_to_df(d[s], split=s))
    out = pd.concat(out, axis=0)
    out['horizon'] = horizons
    out['model'] = [model.name] + [''] * (out.shape[0] - 1)
    out['time_scale'] = [model.time_scale] + [''] * (out.shape[0] - 1)  
    cols = flatten([['time_scale', 'model', 'horizon'], metrics])
    return out[cols]


def flatten(l):
    '''Flattens a list.'''
    return [item for sublist in l for item in sublist]


def is_arrtype(x):
    '''Tests whether an object is an array-type.'''
    if type(x) in [type([0]), type(np.array(0))]:
        return True
    else:
        return False


def split_series(series):
    '''Simple wrapper for breaking the time series into splits.'''
    train = series[series.split == 'train'].reset_index(drop=True)
    val = series[series.split == 'val'].reset_index(drop=True)
    test = series[series.split == 'test'].reset_index(drop=True)
    return train, val, test


def make_series(data, outcome, time_col, geo_col=None):
    '''Simple wrapper for grouping a time series by time and place.'''
    group_cols = [time_col, 'year', 'split']
    sort_cols = [time_col]
    if geo_col:
        group_cols.append(geo_col)
        sort_cols = [geo_col, time_col]
    grouped= data.groupby(group_cols, as_index=False)
    series = grouped[outcome].sum()
    sorted_series = series.sort_values(sort_cols)
    return sorted_series


def run_arima(series,
              outcome, 
              model_types=['AutoARIMA'],
              horizons=[1], 
              time_scale=None):
    '''Runs different ARIMA models on a time series.'''
    splits = split_series(series)
    train, val, test = [s[outcome].values for s in splits]
    mods = {}
    for model_type in model_types:
        mod = getattr(models, model_type)(time_scale=time_scale)
        mod.fit(train)
        if model_type == 'AutoARIMA':
            mod.val_select(val)
        else:
            mod.score(val)
        mod.score(test, 'test', 'val', horizons)
        mods.update({model_type: mod})
    return mods


def stagger(ts, window_length=4, return_y=True, horizon=1):
    '''Breaks down a single time series into its staggered components.'''
    max_length = len(ts)
    n_staggers = max_length - window_length + 1
    start = 0
    stop = window_length
    staggers = []
    for stag in list(range(n_staggers)):
        staggers.append(ts[start:stop])
        start += 1
        stop += 1
    x, y = np.array(staggers), None
    if return_y:
        y = x[horizon:, -1]
        x = x[:-horizon]
    return x, y


def stitch(ts0, ts1, 
           horizon=1, 
           window_length=4,
           recurrent=False,
           expand=False):
    '''Stitches together two time series. Kind of a mess right now.'''
    stag = np.ndim(ts0[0]) == 1
    if stag:
        x0, _ = stagger(ts0, 
                        window_length=window_length, 
                        horizon=horizon, 
                        return_y=True)
        x1, y1  = stagger(ts1, 
                          window_length=window_length,
                          horizon=horizon, 
                          return_y=True)
    else:
        x0, _ = ts0[0], ts0[1]
        x1, y1 = ts1[0], ts1[1]
    if recurrent:
        if expand:
            x0 = np.expand_dims(x0, axis=2)
            x1 = np.expand_dims(x1, axis=2)
        old_end = x0[-1:]
        new_start = x1[0:1]
        window_length = old_end.shape[1]
    else:
        old_end = x0[-1]
        new_start = x1[0]
        window_length = old_end.shape[0]
    
    gap_x, gap_y = stagger(np.concatenate((old_end, new_start)).flatten(),
                           window_length=window_length,
                           horizon=horizon)
    if recurrent:
        gap_x = np.expand_dims(gap_x, axis=2)
    
    gap_y = gap_y.flatten()
    stitch_x = np.concatenate((gap_x, x1))
    stitch_y = np.concatenate((gap_y, y1))
    return stitch_x, stitch_y