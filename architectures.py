import torch
import torch.nn as nn


class LSTMLastOutput(nn.Module):
    def __init__(self,n_inputs=1, hidden_size=50, num_layers=1, dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_iputs = n_inputs
        if num_layers == 1:
            dropout=0
        self.lstm_0 = nn.LSTM(input_size=n_inputs, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm_0(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return torch.unsqueeze(x, dim=-1)
class LSTMWholeOutputs(nn.Module):
    def __init__(self,n_inputs=1, hidden_size=50, num_layers=1, dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_iputs = n_inputs
        if num_layers == 1:
            dropout=0
        self.lstm_0 = nn.LSTM(input_size=n_inputs, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm_0(x)
        x = self.linear(x)
        return torch.unsqueeze(x, dim=-1)