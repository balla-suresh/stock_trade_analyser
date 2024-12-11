import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
from datetime import datetime

from src.lib.tools.data_utils import *
from src.lib.models.model import *
from src.lib.tools.log_utils import LoggerUtils
from src.lib.tools.downloader import Dowloader
from src.lib.tools.file_utils import FileUtils

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = None
interval = '5m'
period = '50d'
is_download = True
logger = LoggerUtils("stock_predictor").get_logger()
logger.info("Started Predicting")

file_utils = FileUtils()
file_utils.clean()
loader = Dowloader(period=period, interval=interval, is_download=is_download)
data = loader.download()
ticker_list = loader.get_ticker_list()

df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')

each_ticker = ticker_list[0]
current_data = file_utils.import_csv(each_ticker)
current_data = current_data.dropna()
current_data = current_data.rename(columns=str.lower)
format = None
if current_data.index.dtype == 'object':
    format = "%Y-%m-%d %H:%M:%S"
    current_data['date'] = pd.to_datetime(current_data.index).strftime(format)
else:
    format = "%Y-%m-%d"
    current_data['date'] = pd.to_datetime(current_data.index).strftime(format)

# close value is stock price at the end of that day
# plt.figure(figsize=(12, 6))
# plt.plot(list(range(len(current_data))), current_data['close'])
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.show()


# normalize dataset and keep track of close value
current_data['date'] = current_data['date'].apply(
    lambda x: string_to_time(x, format))
min_max_dictionary = min_max_dic(current_data)
# print(current_data)

stock_value = current_data[['close']]

# split data into training and testing components
x_train, y_train, x_test, y_test = split_data(stock_value, 20)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

input_size = 1
hidden_dim = 64
num_layers = 3
output_dim = 1
epochs = 100

model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
loss_func = nn.MSELoss(reduction='mean').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss = []
min_loss = np.Inf

for epoch in range(epochs):
    # zero out gradients
    optimizer.zero_grad()

    pred = model(x_train.to(device))
    loss = loss_func(pred, y_train.to(device))
    train_loss.append(loss.item())
    logger.info('Epoch {},    Loss: {:.10f}\n'.format(epoch, loss.item()))

    # keep track of lowest loss and save as best model
    if loss.item() < min_loss:
        logger.info(
            '     New Minimum Loss: {:.10f} ----> {:.10f}\n'.format(min_loss, loss.item()))
        min_loss = loss.item()
        torch.save(model.state_dict(), 'predictions/' + each_ticker + '.pt')

    # back propogate
    loss.backward()
    optimizer.step()

# visualize loss throughout training
# plt.figure(figsize=(12, 6))
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.plot(list(range(epochs)), train_loss)


best_model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
best_model.load_state_dict(torch.load('predictions/' + each_ticker + '.pt'))

# get predictions
lstm_predictions = model(x_test.to(device))
test_loss = loss_func(lstm_predictions, y_test.to(device))
lstm_predictions = lstm_predictions.squeeze().tolist()

min = min_max_dictionary['close'][0]
max = min_max_dictionary['close'][1]

# reverse normalize predictions and dataset values
lstm_predictions = [reverse_normalize(x, min, max)
                    for x in np.array(lstm_predictions)]

stock_value = [reverse_normalize(x, min, max) for x in np.array(stock_value)]

plt.figure(figsize=(12, 6))
plt.plot([i for i in range(x_train.size(0))],
         stock_value[:x_train.size(0)], color='b', label='trained values')

# plot test range and predictions by the GRU
time_values_actual = list(range(x_train.size(0), len(stock_value)))
time_values_pred = list(
    range(x_train.size(0), x_train.size(0) + y_test.size(0)))
plt.xlabel('Time')
plt.ylabel('Stock Values')
plt.plot(time_values_actual,
         stock_value[-len(time_values_actual):], color='r', label='actual values')
plt.plot(time_values_pred[3:], lstm_predictions[3:],
         color='g', linewidth=2, label='predicted values')

plt.show()

print(lstm_predictions)
