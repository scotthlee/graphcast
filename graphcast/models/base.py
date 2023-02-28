'''Base class for time series models.'''


class TimeSeriesModel():
    def __init__(self,
                 results=None,
                 gof=None,
                 order=None, 
                 time_scale=None,
                 name=None,
                 model_type=None):
        self.results = results
        self.gof = gof
        self.forecasts = {}
        self.scores = {}
        self.order = order
        self.time_scale = time_scale
        self.indices = {}
        self.name = name
        self.model_type = model_type
        return
