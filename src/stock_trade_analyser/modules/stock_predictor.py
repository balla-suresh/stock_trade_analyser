import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import operator
from datetime import timedelta

from stock_trade_analyser.tools.data_utils import *
from stock_trade_analyser.models.model import *
from stock_trade_analyser.tools.log_utils import LoggerUtils
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.tools.file_utils import FileUtils, parse_ticker_file_arg

import matplotlib.pyplot as plt
import multiprocessing

import json

# Parsed at module top-level so multiprocessing `spawn` workers, which re-import
# this module with the parent's argv, see the same `--ticker-file` value.
_TICKER_FILE_ARG = parse_ticker_file_arg(
    prog="stock_predictor",
    description="Run the LSTM-based stock price predictor.",
)

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json'), 'r') as f:
    config = json.load(f)

# normalize
scaler = Normalizer()

# model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = LSTM(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]
["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1)
model = model.to(config["training"]["device"])


def predict(each_ticker):
    logger = LoggerUtils("stock_predictor").get_logger()
    file_utils = FileUtils(
        data_type=config["download"]["data_type"],
        ticker_file=_TICKER_FILE_ARG,
    )
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    current_data.index = pd.to_datetime(current_data.index)
    data_close_price = np.array(current_data['close'])

    # normalize
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    prepared = prepare_data(normalized_data_close_price, config)
    if prepared is None:
        logger.warning(f"{each_ticker}: Not enough data points for window size {config['data']['window_size']}")
        return {each_ticker: []}
        
    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepared

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
    best_model_state = None
    if config["download"]["is_download"]:
        min_loss = np.inf
        epochs_no_improve = 0
        patience = 15  # Early stopping patience
        
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
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                logger.debug(f"{each_ticker}: Early stopping triggered at epoch {epoch + 1}")
                break

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    best_model = LSTM(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]
    ["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1)
    
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state)
    else:
        # Fallback if not training (e.g. is_download is False) and we still want to try loading from disk
        pt_path = file_utils.get_predictions() + '/' + file_utils.get_data_type() + '/' + each_ticker + '.pt'
        if os.path.exists(pt_path):
            best_model.load_state_dict(torch.load(pt_path))
        else:
            best_model.load_state_dict(model.state_dict())

    best_model.eval()

    # predict on the unseen data, tomorrow's price

    best_model.eval()
    x_future = 5
    predictions = np.array([])
    dicts = []
    curr_date = current_data.index[-1]
    
    # We must ensure data_x_unseen is a numpy array before tensor conversion
    if not isinstance(data_x_unseen, np.ndarray):
        data_x_unseen = np.array(data_x_unseen)
        
    for i in range(x_future):
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(
            0).unsqueeze(2)  # this is the data type and shape required, [batch, sequence, feature]

        prediction = best_model(x)
        prediction = prediction.cpu().detach().numpy()
        data_x_unseen = data_x_unseen[1:]
        data_x_unseen = np.append(data_x_unseen, prediction)
        prediction = scaler.inverse_transform(prediction)[0]
        # Make sure we extract the scalar value if it's an array
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        curr_date = curr_date + timedelta(days=1)
        dicts.append({'Predictions': float(prediction), "Date": str(curr_date)})

    logger.debug(
        f"{each_ticker}: Predicted close price of the next days: {dicts}")

    return {each_ticker : dicts}


if __name__ == '__main__':
    logger = LoggerUtils("stock_predictor").get_logger()
    file_utils = FileUtils(
        data_type=config["download"]["data_type"],
        ticker_file=_TICKER_FILE_ARG,
    )
    logger.info("Started Predicting")
    logger.info(f"Using ticker file: {file_utils.ticker_file}")
    file_utils.clean()
    loader = Downloader(period=config["download"]["period"], interval=config["download"].get("interval", "1d"),
                    is_download=config["download"]["is_download"], file_utils=file_utils)

    data = loader.download()
    ticker_list = loader.get_ticker_list()

    # df = pd.DataFrame(ticker_list, columns=['symbol'])
    # df = df.set_index('symbol')
    
    # We must unpack the ticker_list properly if it's a list of dicts
    clean_ticker_list = []
    for t in ticker_list:
        if isinstance(t, dict):
            clean_ticker_list.append(t['symbol'])
        else:
            clean_ticker_list.append(t)
            
    with multiprocessing.Pool() as pool:
        outputs_async = pool.map_async(predict, clean_ticker_list)
        outputs = outputs_async.get()
        
    # outputs is a list of dicts, e.g. [{'AAPL': [...]}, {'MSFT': [...]}]
    # We need to merge them into a single dict before writing to JSON
    merged_outputs = {}
    for output in outputs:
        if output and isinstance(output, dict):
            # Ensure all float32/float64 are converted to standard float for JSON serialization
            for ticker, preds in output.items():
                if not preds:
                    continue
                clean_preds = []
                for p in preds:
                    # Handle both dict and float formats
                    if isinstance(p, dict):
                        clean_preds.append({
                            'Predictions': float(p['Predictions']),
                            'Date': str(p['Date'])
                        })
                    else:
                        clean_preds.append(float(p))
                merged_outputs[ticker] = clean_preds
                
    if not merged_outputs:
        logger.warning("merged_outputs is empty! Check if predict() is returning data.")
            
    # logger.info("Output: {}".format(outputs))
    # logger.info(json.dumps(outputs, indent = 3))
    
    # Write directly to predictions folder
    predictions_dir = file_utils.get_predictions()
    os.makedirs(predictions_dir, exist_ok=True)
    path = os.path.join(predictions_dir, 'final.json')
    
    with open(path, 'w') as f:
        json.dump(merged_outputs, f, indent=4)
    logger.info(f"Finished Predicting, saved to {path}")
