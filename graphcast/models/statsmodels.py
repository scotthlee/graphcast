import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as stm
import itertools
import tensorflow as tf
import keras_tuner

from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from .. import metrics, tools
from .base import TimeSeriesModel

class StatsmodelsModel(TimeSeriesModel):
    def __init__(self,
                 results=None,
                 gof=None,
                 order=None,
                 time_scale=None,
                 name=None,
                 model=None):
        super().__init__(results=results,
                         gof=gof,
                         order=order,
                         time_scale=time_scale,
                         name=name,
                         model_type='statsmodels')
    
    def forecast(self,
                 rfp,
                 trend, 
                 steps_ahead=1, 
                 frequency=1,
                 refit=True):
        res = self.results[rfp]
        idx, cast = [], []
        start, end = 0, steps_ahead
        max_steps = len(trend) - steps_ahead + 1
        steps = max_steps // frequency
        for step in range(steps):
            f = res.forecast(steps_ahead)[-1]
            idx.append(end - 1)
            cast.append(f)
            res = res.append(trend[start:start+frequency], refit=refit)
            start += frequency
            end += frequency
        cast = np.array(cast).flatten().astype(int)
        return idx, cast, res


class LastValue(StatsmodelsModel):
    def __init__(self, time_scale=None):
        super().__init__(time_scale=time_scale,
                         name='LastValue')
    
    def score(self,
              trend, 
              steps_ahead=1,
              split='test'):
        pred = trend[:-steps_ahead]
        scores = metrics.score_report(trend[steps_ahead:], pred)
        self.scores.update({split: scores})
        self.forecasts.update({split: pred})
        return scores
        

class RandomWalk(StatsmodelsModel):
    def __init__(self, time_scale=None):
        super().__init__(time_scale=time_scale,
                         name='RandomWalk')
    
    def fit(self,
            trend,
            cov_type='robust'):
        self.model = ARIMA(trend, order=(0, 1, 0))
        self.results = {'train': self.model.fit(cov_type=cov_type)}
        return
     
    def score(self,
             trend,
             split='val',
             res_for_pred='train',
             steps_ahead=1,
             frequency=1,
             refit=True):
        if not tools.is_arrtype(steps_ahead):
            steps_ahead = [steps_ahead]
        max_horizon = np.max(steps_ahead)
        for i, stp in enumerate(steps_ahead):
            if split != 'val':
                s = split + '_' + str(stp)
            else:
                s = split
            start = max_horizon - stp
            RFP = res_for_pred
            idx, cast, res = self.forecast(RFP,
                                           trend=trend,
                                           steps_ahead=stp,
                                           frequency=frequency,
                                           refit=refit)
            idx, cast = idx[start:], cast[start:]
            scores = metrics.score_report(trend[idx], cast)
            self.indices.update({s: idx})
            self.scores.update({s: scores})
            self.forecasts.update({s: cast})
            self.results.update({s: res})
        return scores
        

class AutoARIMA(StatsmodelsModel):
    def __init__(self, time_scale=None):
        super().__init__(name='auto_arima',
                         time_scale=time_scale)
    
    def fit(self,
            trend,
            criterion='aic',
            cov_type='robust',
            order_maxes=[3, 2, 3],
            top_n=5):
        ps = np.arange(order_maxes[0] + 1)
        ds = np.arange(order_maxes[1] + 1)
        qs = np.arange(order_maxes[2] + 1)
        orders = list(itertools.product(ps, ds, qs))
        wraps = []
        for order in orders:
            mod = ARIMA(trend, order=order)
            res = mod.fit(cov_type=cov_type)
            res_dict = {'train': mod.fit(cov_type=cov_type)}
            gof = getattr(res, criterion)
            wrap = StatsmodelsModel(model=mod,
                                    results=res_dict,
                                    order=order,
                                    gof=gof)
            wraps.append(wrap)
        gofs = [m.gof for m in wraps]
        sorted = np.argsort(gofs)
        self.mods = [wraps[i] for i in sorted[:top_n]]
        self.mod = wraps[0]
        self.results = wraps[0].results
        return
    
    def val_select(self,
                   trend,
                   steps_ahead=1,
                   frequency=1,
                   refit=True,
                   metric='rmse'):
        self.val_trend = trend
        for mod in self.mods:
            train_res = mod.results['train']
            idx, cast, res = self.forecast(rfp='train',
                                           trend=trend,
                                           steps_ahead=steps_ahead,
                                           frequency=frequency,
                                           refit=refit)
            scores = metrics.score_report(trend[idx], cast)
            mod.scores = {'val': scores}
            mod.forecasts = {'val': cast}
            mod.results.update({'val': res})
        min_score = np.argmin([m.scores['val'][metric] 
                               for m in self.mods])
        best_mod = self.mods[min_score]
        self.scores.update(best_mod.scores)
        self.forecasts.update(best_mod.forecasts)
        self.results.update(best_mod.results)
        self.mod = best_mod
        return
    
    def score(self,
              trend,
              split='test',
              res_for_pred='val',
              steps_ahead=1,
              frequency=1,
              refit=True):
        if not tools.is_arrtype(steps_ahead):
            steps_ahead = [steps_ahead]
        max_horizon = np.max(steps_ahead)
        for i, stp in enumerate(steps_ahead):
            start = max_horizon - stp
            s = split + '_' + str(stp)
            idx, cast, res = self.forecast(rfp=res_for_pred,
                                           trend=trend,
                                           steps_ahead=stp,
                                           frequency=frequency,
                                           refit=refit)
            idx, cast = idx[start:], cast[start:]
            scores = metrics.score_report(trend[idx], cast)
            self.indices.update({s: idx})
            self.scores.update({s: scores})
            self.forecasts.update({s: cast})
            self.results.update({s: res})
        return
