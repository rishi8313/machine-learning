import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocessor():
    
    def __init__(self, df):
        self.df = df
    
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
        
    def Run(self):
        self.df = self.df[(self.df['VehicleSpeed'] >= -1) & (self.df['TotalAcc'] >= -11)]
        self.df = pd.get_dummies(self.df, prefix='ShiftNumber',columns = ["ShiftNumber"],drop_first=False)
        values = self.df.drop('Time',axis=1).values
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, 1, 1)
        # drop columns we don't want to predict
        reframed.drop(reframed.columns[np.arange(21,36)], axis=1, inplace=True)
        return reframed

