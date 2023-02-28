import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras_tuner

from matplotlib import pyplot as plt

from .. import metrics, tools
from .base import TimeSeriesModel


class KerasModel(TimeSeriesModel):
    def __init__(self):
        '''A light wrapper class for Keras models that allows for both 
        standard training procedures and hyperparameter optimization via 
        Keras Tuner.
        '''
        super().__init__(model_type='Keras')
        self.data = {'train': None, 
                     'val': None, 
                     'test': None}
        self.models = {'train': None,
                       'val:': None,
                       'test': None}
        self.tuner = None
        self.recurrent = False
        return
    
    def build(self):
        pass
    
    def build_hp(self):
        pass
    
    def build_tuner(self, 
                    search_type='Hyperband', 
                    objective='val_loss',
                    **kwargs):
        tuner = getattr(keras_tuner, search_type)(
            hypermodel=self.build_hp,
            objective=objective,
            **locals()['kwargs']
        )
        self.tuner = tuner 
    
    def compile(self,
                learning_rate=0.001,
                loss='MeanSquaredError',
                optimizer='Adam',
                metrics=['MeanAbsoluteError']):
        loss = getattr(tf.keras.losses, loss)()
        optimizer = getattr(tf.keras.optimizers, optimizer)(
            learning_rate=learning_rate
        )
        metrics = [getattr(tf.keras.metrics, m)() for m in metrics]
        self.models['train'].compile(loss=loss,
                                     optimizer=optimizer,
                                     metrics=metrics)
    
    def score(self,
              x, y,
              steps_ahead=1,
              model='val',
              base_data='train',
              overlap=True,
              refit=True,
              **kwargs):
        expand = (self.recurrent & (len(x.shape) < 3))
        if overlap:
            old_ts = (self.data[base_data]['x'], self.data[base_data]['y'])
            val_x, val_y = tools.stitch(ts0=old_ts,
                                        ts1=(x, y),
                                        expand=expand,
                                        horizon=steps_ahead,
                                        recurrent=self.recurrent)
        else:
            val_x, val_y = x, y
        
        cast = []
        for ts in val_x:
            if self.recurrent:
                ts = np.expand_dims(ts, axis=0)
            else:
                ts = ts.reshape(1, -1)
            _, pred = self.forecast(x=ts,
                                    steps_ahead=steps_ahead, 
                                    model=model, 
                                    verbose=0)
            cast.append(pred)
        
        cast = np.array(cast).flatten()
        scores = metrics.score_report(val_y, cast)
        self.forecasts.update({model: cast})
        self.scores.update({model: scores})
        
    def __fit(self, x, y, model='train', **kwargs):
        mod = self.models[model]
        mod.fit(x, y,
                batch_size=batch_size,
                **locals()['kwargs'])
        self.models.update({model: mod})
    
    def fit_tuner(self, x, y,
                  **kwargs):
        if self.tuner is None:
            print('Please build a tuner with .build_tuner() before fitting.')
            return
        self.tuner.search(x, y, **locals()['kwargs'])
    
    def forecast(self, x,
                 steps_ahead=1,
                 model='val', 
                 **kwargs):
        cast = []	
        for step in range(steps_ahead):
            pred = self.__predict(x, model=model, **locals()['kwargs'])
            pred = pred.round()
            cast.append(pred.flatten())
            if self.recurrent:
                x = np.concatenate((x[:, 1:, :], 
                                    pred.reshape(-1, 1, 1)), axis=1)
            else:
                x = np.concatenate((x[:, 1:], pred), axis=1)
         
        cast = np.array(cast).flatten()                       
        return cast, cast[-1]
    
    def __predict(self, x, model='train', **kwargs):
        '''A wrapper for the predict method of underlying Keras model.'''
        return self.models[model].predict(x, **locals()['kwargs'])
    
    def summary(self):
        self.models['train'].summary()
        
    def train(self, x, y,
              validation_data=None,
              save_path=None,
              **kwargs):
        self.models['train'].fit(x, y, 
                                 validation_data=validation_data,
                                 **locals()['kwargs'])
        self.models['train'].save(save_path)
        self.models['val'] = tf.keras.models.load_model(save_path)
        self.data.update({'train': {'x': x, 
                                    'y': y},
                          'val': {'x': validation_data[0], 
                                  'y': validation_data[1]}})


class Dense(KerasModel):
    def __init__(self):
        super().__init__()
        self.name = 'Dense'
    
    def build(self,
              units=64,
              num_layers=2, 
              activation='relu'):
        model = tf.keras.Sequential()
        for l in range(num_layers):
            model.add(tf.keras.layers.Dense(units=units,
                                            activation=activation))
        model.add(tf.keras.layers.Dense(1))
        self.models['train'] = model
    
    def build_hp(self, hp,
                 loss='MeanSquaredError',
                 optimizer='Adam',
                 metrics=['MeanAbsoluteError']):
        num_layers = hp.Int('layers', min_value=1, max_value=5, step=1)
        units = hp.Int('units', min_value=16, max_value=64, step=8)
        self.build(units=units, 
                   num_layers=num_layers)
        self.compile(loss=loss,
                      optimizer=optimizer, 
                      metrics=metrics)
        return self.results['train']


class LSTM(KerasModel):
    def __init__(self, sequence_length=4):
        super().__init__()
        self.recurrent = True
        self.name = 'LSTM'
        self.sequence_length = sequence_length
        return
    
    def adapt_norm(self, x):
        self.model.layers[0].adapt(x)
    
    def build(self,
              units=16,
              num_features=1,
              num_layers=1,
              dropout=0.2,
              unit_decrease_factor=1,
              activation='tanh',
              recurrent_activation='relu',
              recurrent_regularizer=None,
              kernel_regularizer=None,
              norm=True,
              **kwargs):
        # Handling None inputs for the HP tuner
        act = recurrent_activation
        if recurrent_regularizer == 'None':
            recurrent_regularizer = None
        if kernel_regularizer == 'None':
            kernel_regularizer = None
        if activation == 'None':
            activation = None
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.sequence_length,
                                                          num_features)))
        if norm:
            model.add(tf.keras.layers.Normalization(axis=-1))
        for l in range(num_layers):
            layer_units = int(units * (1 / (unit_decrease_factor ** (l + 1))))
            if l == num_layers - 1:
                sequences = False
            else:
                sequences = True
            model.add(tf.keras.layers.LSTM(units=layer_units,
                                           dropout=dropout,
                                           recurrent_activation=act,
                                           return_sequences=sequences,
                                           **locals()['kwargs']))
        model.add(tf.keras.layers.Dense(1))
        self.models['train'] = model
    
    def build_hp(self, hp,
                 loss='MeanSquaredError',
                 optimizer='Adam',
                 metrics=['MeanAbsoluteError']):
        num_layers = hp.Int('layers', min_value=1, max_value=2, step=1)
        unit_decrease = hp.Int('unit_decrease', 
                               min_value=1, 
                               max_value=2, 
                               step=1)
        units = hp.Int('units', min_value=4, max_value=32, step=8)
        drop = hp.Float('dropout', 
                        min_value=0, max_value=0.8, step=.1)
        r_drop = hp.Float('recurrent_dropout', 
                          min_value=0, 
                          max_value=0.8, 
                          step=.1)
        k_reg = hp.Choice('kernel_regularizer', 
                          values=['None', 'l1', 'l2'])
        r_reg = hp.Choice('recurrent_regularizer',
                          values=['None', 'l1', 'l2'])
        activation = hp.Choice('activation',
                               values=['None', 'relu', 'tanh'])
        r_activation = hp.Choice('recurrent_activation',
                                 values=['relu', 'selu', 'elu'])
        learning_rate = hp.Choice('learning_rate',
                                 values=[1e-2, 1e-3, 1e-4, 1e-5])
        self.build(units=units,
                   activation=activation,
                   recurrent_activation=r_activation, 
                   kernel_regularizer=k_reg,
                   recurrent_regularizer=r_reg,
                   num_layers=num_layers,
                   dropout=drop,
                   recurrent_dropout=r_drop,
                   unit_decrease_factor=unit_decrease)
        self.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics,
                     learning_rate=learning_rate)
        return self.models['train']