from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import *
import pandas as pd
import numpy as np


class MLP:
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), MLPClassifier(
            random_state=0, shuffle=False))

    def setup(self, df):
        df["Diff"] = df.close.diff()
        df["SMA_2"] = df.close.rolling(2).mean()
        df["Force_Index"] = df["close"] * df["volume"]
        df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
        df = df.drop(
            ["open", "high", "low", "close", "volume", "Diff", "adj close"],
            axis=1,
        ).dropna()
        return df
        
    def evalute(self, df):
        df = self.setup(df)
        # print(df)
        X = df.drop(["y"], axis=1).values
        y = df["y"].values
        # print(f"x------{X}")
        # print(f"y------{y}")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

        self.clf.fit(
            X_train,
            y_train,
        )
        y_pred = self.clf.predict(X_test, verbose=0)
        # print(f"y_pred========{y_pred}")
        # print(f"y_test========{y_test}")
        return y_test, y_pred


class LSTMmodel:
    def __init__(self):
        self.lstm = None
        self.scaler = None
    
    def setup(self, df):
        self.lstm = Sequential()
        self.scaler = MinMaxScaler()
        df["Diff"] = df.close.diff()
        df["SMA_2"] = df.close.rolling(2).mean()
        df["Force_Index"] = df["close"] * df["volume"]
        df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
        df = df.drop(
            ["open", "high", "low", "close", "volume", "Diff", "adj close"],
            axis=1,
        ).dropna()
        return df

    def evalute(self, df):
        df = self.setup(df)
        # print(df)
        X = StandardScaler().fit_transform(df.drop(["y"], axis=1))
        y = df["y"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

        self.lstm.add(LSTM(32, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=False))
        self.lstm.add(Dense(1))
        self.lstm.compile(loss='binary_crossentropy', optimizer='adam')
        history = self.lstm.fit(X_train[:, :, np.newaxis], y_train, epochs=100, verbose=0)
        y_pred = self.lstm.predict(X_test, verbose=0)
        # print(f"y_pred========{y_pred}")
        # print(f"y_test========{y_test}")
        return y_test, y_pred  > 0.5


class GRUmodel:
    def __init__(self):
        self.model = None
    
    def setup(self, df):
        self.model = Sequential()
        df["Diff"] = df.close.diff()
        df["SMA_2"] = df.close.rolling(2).mean()
        df["Force_Index"] = df["close"] * df["volume"]
        df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)
        df = df.drop(
            ["open", "high", "low", "close", "volume", "Diff", "adj close"],
            axis=1,
        ).dropna()
        return df
    
    def evalute(self, df):
        df = self.setup(df)
        # print(df)
        X = StandardScaler().fit_transform(df.drop(["y"], axis=1))
        y = df["y"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )
        
        self.model.add(GRU(2, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
        self.model.fit(X_train[:, :, np.newaxis], y_train, epochs=100, verbose=0)
        y_pred = self.model.predict(X_test[:, :, np.newaxis], verbose=0)
        # print(f"y_pred========{y_pred}")
        # print(f"y_test========{y_test}")
        return y_test, y_pred  > 0.5
    
