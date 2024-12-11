import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import operator
from datetime import timedelta

from src.lib.tools.data_utils import *
from src.lib.models.model import *
from src.lib.tools.log_utils import LoggerUtils
from src.lib.tools.downloader import Downloader
from src.lib.tools.file_utils import FileUtils

import matplotlib.pyplot as plt
import multiprocessing

import json

config = {
    "download": {
        "interval": "15m",
        "period": "1mo",
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
loader = Downloader(period=config["download"]["period"], interval=config["download"].get("interval"),
                    is_download=config["download"]["is_download"], file_utils=file_utils)

# normalize
scaler = Normalizer()

# model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = LSTM(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]
["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1)
model = model.to(config["training"]["device"])


def predict(each_ticker):
    # each_ticker = ticker_list[0]
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    current_data.index = pd.to_datetime(current_data.index)
    data_close_price = np.array(current_data['close'])

    # normalize
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(
        normalized_data_close_price, config)

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    logger.debug(
        f"{each_ticker}: Train data shape {dataset_train.x.shape} {dataset_train.y.shape}")
    logger.debug(
        f"{each_ticker}: Validation data shape {dataset_val.x.shape} {dataset_val.y.shape}")

    # create `DataLoader`
    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(
    ), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

    # begin training
    if not config["download"]["is_download"]:
        min_loss = np.Inf
        for epoch in range(config["training"]["num_epoch"]):
            loss_train, lr_train = run_epoch(
                model, optimizer, criterion, scheduler, config, train_dataloader, is_training=True)
            loss_val, lr_val = run_epoch(
                model, optimizer, criterion, scheduler, config, val_dataloader)
            scheduler.step()

            # keep track of lowest loss and save as best model
            logger.debug(f"{each_ticker}: " + "Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}"
                         .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
            if loss_train < min_loss:
                logger.debug(
                    '     New Minimum Loss: {:.10f} ----> {:.10f}\n'.format(min_loss, loss_train))
                min_loss = loss_train
                torch.save(model.state_dict(),
                           file_utils.get_predictions() + '/' + each_ticker + '.pt')

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    best_model = LSTM(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]
    ["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1)
    best_model.load_state_dict(torch.load(
        file_utils.get_predictions() + '/' + each_ticker + '.pt'))

    best_model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        out = best_model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = best_model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # predict on the unseen data, tomorrow's price

    best_model.eval()
    x_future = 5
    predictions = np.array([])
    dicts = []
    curr_date = current_data.index[-1]
    for i in range(x_future):
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(
            0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]

        prediction = best_model(x)
        prediction = prediction.cpu().detach().numpy()
        data_x_unseen = data_x_unseen[1:]
        data_x_unseen = np.append(data_x_unseen, prediction)
        prediction = scaler.inverse_transform(prediction)[0]
        curr_date = curr_date + timedelta(days=1)
        dicts.append({'Predictions': prediction, "Date": str(curr_date)})

    logger.debug(
        f"{each_ticker}: Predicted close price of the next days: {dicts}")

    return {each_ticker : dicts}


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
