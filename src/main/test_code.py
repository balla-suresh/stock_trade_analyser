import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import operator
from datetime import timedelta

from src.lib.tools.data_utils import *
from src.lib.models.model import *
from src.lib.tools.log_utils import LoggerUtils
from src.lib.tools.downloader import Dowloader
from src.lib.tools.file_utils import FileUtils

import multiprocessing

import json

config = {
    "download": {
        # "interval": "1d",
        "period": "2y",
        "is_download": False
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 3,
        "lstm_size": 64,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

logger = None
file_utils = None
logger = LoggerUtils("stock_predictor").get_logger()
file_utils = FileUtils()
loader = Dowloader(period=config["download"]["period"], interval=config["download"].get("interval"),
                   is_download=config["download"]["is_download"], file_utils=file_utils)

# normalize
scaler = Normalizer()

# model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = LSTM(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]
["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1)
model = model.to(config["training"]["device"])






# Define lambda function
def _lambda(n):
    if n <= 37:
        switcher = {
            2: math.pi / 3.0,
            3: math.atan(math.sqrt(0.6)),
            4: 2.153460564 / n,
            5: 1.923796031 / n,
            6: 1.915022415 / n,
            7: 1.909786299 / n,
            8: 1.906409362 / n,
            9: 1.904103844 / n,
            10: 1.902459533 / n,
            11: 1.901245508 / n,
            12: 1.900323600 / n,
            13: 1.899607018 / n,
            14: 1.899038987 / n,
            15: 1.898581041 / n,
            16: 1.898206498 / n,
            17: 1.897896254 / n,
            18: 1.897636390 / n,
            19: 1.897416484 / n,
            20: 1.897228842 / n,
            21: 1.897067382 / n,
            22: 1.896927473 / n,
            23: 1.896805427 / n,
            24: 1.896698359 / n,
            25: 1.896603866 / n,
            26: 1.896520032 / n,
            27: 1.896445477 / n,
            28: 1.896378692 / n,
            29: 1.896318725 / n,
            30: 1.896264646 / n,
            31: 1.896215693 / n,
            32: 1.896171301 / n,
            33: 1.896130841 / n,
            34: 1.896094060 / n,
            35: 1.896060192 / n,
            36: 1.896029169 / n,
            37: 1.896000584 / n
        }
        w = switcher.get(n, math.pi / 2 / n)
    else:
        w = math.pi / 2 / n

    lamb = 0.0625 / math.pow(math.sin(w), 4)
    return lamb


# Hodrick-Prescott filter function
def _HPFilter(x, y, lamb, per):
    a = np.zeros(per)
    b = np.zeros(per)
    c = np.zeros(per)

    a[0] = 1.0 + lamb
    b[0] = -2.0 * lamb
    c[0] = lamb

    for i in range(1, per - 3):
        a[i] = 6.0 * lamb + 1.0
        b[i] = -4.0 * lamb
        c[i] = lamb

    a[1] = 5.0 * lamb + 1.0
    a[per - 1] = 1.0 + lamb
    a[per - 2] = 5.0 * lamb + 1.0
    b[per - 2] = -2.0 * lamb

    H1, H2, H3, H4, H5 = 0., 0., 0., 0., 0.
    HH1, HH2, HH3, HH5 = 0., 0., 0., 0.
    HB, HC, Z = 0., 0., 0.

    for i in range(per):
        Z = a[i] - H4 * H1 - HH5 * HH2
        if Z == 0:
            break

        HB = b[i]
        HH1 = H1
        H1 = (HB - H4 * H2) / Z
        b[i] = H1

        HC = c[i]
        HH2 = H2
        H2 = HC / Z
        c[i] = H2

        a[i] = (x[i] - HH3 * HH5 - H3 * H4) / Z
        HH3 = H3
        H3 = a[i]
        H4 = HB - H5 * HH1
        HH5 = H5
        H5 = HC

    H2 = 0
    H1 = a[per - 1]
    y[per - 1] = H1

    for i in range(per - 2, -1, -1):
        y[i] = a[i] - b[i] * H1 - c[i] * H2
        H2 = H1
        H1 = y[i]


def predict(each_ticker):
    # each_ticker = ticker_list[0]
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    current_data.index = pd.to_datetime(current_data.index)
    src = np.array(current_data['close'])

    # Inputs (example values)
    LastBar = 0
    PastBars = src.shape[0]
    FutBars = 100

    # Example source data (close prices)
    # src = np.random.randn(5000)  # Replace with actual data

    x = np.zeros(PastBars)
    y = np.zeros(PastBars)
    fv = np.zeros(PastBars)

    # Hodrick-Prescott filtering and extrapolation
    if PastBars > FutBars:
        for i in range(PastBars):
            x[i] = src[i + LastBar]

        fv[FutBars] = src[LastBar]
        sum_val = src[LastBar]

        Method = 1  # Example method choice
        if Method == 2:
            n = 2 * FutBars + 1
            _HPFilter(x, y, _lambda(n), PastBars)
            for i in range(1, n - 1):
                sum_val += src[i + LastBar]

        for i in range(1, FutBars + 1):
            if Method == 1:
                n = 2 * i + 1
                _HPFilter(x, y, _lambda(n), PastBars)
                sum_val += src[i + LastBar]
                fv[FutBars - i] = n * y[0] - sum_val
                sum_val = n * y[0]
            else:
                fv[FutBars - i] = n * y[FutBars - i] - sum_val
                sum_val += fv[FutBars - i] - src[n - 1 + LastBar - i]

    # Plotting example
    plt.plot(src[LastBar:], label='Source Data')
    plt.plot(fv[:FutBars], label='Extrapolated Data')
    print(fv)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    logger.info("Started Predicting")
    file_utils.clean()
    data = loader.download()
    ticker_list = loader.get_ticker_list()

    # df = pd.DataFrame(ticker_list, columns=['symbol'])
    # df = df.set_index('symbol')
    pool = multiprocessing.Pool()
    outputs_async = pool.map_async(predict, ticker_list)
    outputs = outputs_async.get()
    # logger.info("Output: {}".format(outputs))
    # logger.info(json.dumps(outputs, indent = 3))
    file_utils.write_json(outputs, "final.json")
    logger.info("Finished Predicting")
