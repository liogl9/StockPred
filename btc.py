import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tqdm import tqdm

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

class LSTMModel(nn.Module):
    def __init__(self,hidden_size=50, num_layers=1, dropout = 0):
        super().__init__()
        if num_layers ==1:
            dropout=0
        self.lstm_0 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout)
        # self.lstm_1 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, return_sequences=True,droput=0.2)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm_0(x)
        # x = x[:, -4, :]
        x = self.linear(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Setup complete. Using torch {torch.__version__}({device})")
torch.manual_seed(42)

loopback_list = [5,15,30,60,100]
period_list = [1,2,5]

for period in period_list:
    for lookback in loopback_list:
        PATH = os.path.join(os.getcwd(),"models","{}daysLoopback".format(lookback),"btc{}y".format(period))
        try:
            os.makedirs(PATH)
        except:
            None
        btc = yf.Ticker("BTC-USD")
        historico_btc = btc.history(period="{}y".format(period))

        stock_price = historico_btc.iloc[:,1:2].values.astype('float32')
        stock_price = stock_price.reshape((-1,1))

        sc = MinMaxScaler(feature_range=(0,1))

        training_set_scaled = sc.fit_transform(stock_price)
        print("Min: ",stock_price.min())
        print("Max: ", stock_price.max())
        plt.show()


        # train-val split for time series
        train_size = int(len(training_set_scaled) * 0.67)
        val_size = len(training_set_scaled) - train_size
        train, val = training_set_scaled[:train_size], training_set_scaled[train_size:]
        X_train, y_train = create_dataset(train, lookback=lookback)
        X_val, y_val = create_dataset(val, lookback=lookback)
        train_min = stock_price[:train_size,:].min()
        train_max = stock_price[:train_size,:].max()
        val_min = stock_price[train_size:,:].min()
        val_max = stock_price[train_size:,:].max()
        print(X_train.shape, y_train.shape)
        print(X_val.shape, y_val.shape)
        print("train Min: ",stock_price[:train_size,:].min())
        print("val Min: ",stock_price[train_size:,:].min())
        print("train max: ",stock_price[:train_size,:].max())
        print("val max: ",stock_price[train_size:,:].max())
        sc_train = MinMaxScaler(feature_range=(train_min, train_max))
        sc_val = MinMaxScaler(feature_range=(val_min, val_max))

        model = LSTMModel(10,1,0).to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.0005)
        loss_fn = nn.MSELoss()
        train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=64)
        val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=64)
        n_epochs = 2000
        best_loss = 100
        print("Train start for {} loopback and {} period".format(lookback,period))
        for epoch in tqdm(range(n_epochs)):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validation
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    X_train_gpu = X_train.to(device)
                    X_val_gpu = X_val.to(device)
                    y_pred_train = model(X_train_gpu)
                    train_rmse = np.sqrt(loss_fn(y_pred_train.cpu(), y_train))
                    y_pred_val = model(X_val_gpu)
                    val_rmse = np.sqrt(loss_fn(y_pred_val.cpu(), y_val))
                    # shift train predictions for plotting
                    train_plot = np.ones_like(training_set_scaled) * np.nan
                    # train_plot[lookback:train_size] = sc_train.fit_transform(y_pred_train.cpu()[:, -1, :])
                    train_plot[lookback:train_size] = y_pred_train.cpu()[:, -1, :]
                    # shift val predictions for plotting
                    val_plot = np.ones_like(training_set_scaled) * np.nan
                    # val_plot[train_size+lookback:len(training_set_scaled)] = sc_val.fit_transform(y_pred_val.cpu()[:, -1, :])
                    val_plot[train_size+lookback:len(training_set_scaled)] = y_pred_val.cpu()[:, -1, :]
                    # plot
                    plt.plot(training_set_scaled, c='b')
                    plt.plot(train_plot, c='r')
                    plt.plot(val_plot, c='g')
                    if val_rmse < best_loss:
                        best_loss = val_rmse
                        torch.save(model.state_dict(), os.path.join(PATH,"best_model.pt"))
                        plt.savefig(os.path.join(PATH,"result.png"))
                        print("New Best model saved")
                    plt.close()

                print("Epoch %d: train RMSE %.4f, val RMSE %.4f, best val %.4f" % (epoch, train_rmse, val_rmse, best_loss))