import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLPWithDropoutBN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu', dropout_rate=0.5):
        super().__init__()
        layers = []
        last_dim = input_dim
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu', dropout_rate=0.5):
        super().__init__()
        layers = []
        last_dim = input_dim
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dim=128, num_layers=1, use_layernorm=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = hn[-1]  # [batch, hidden_dim]
        if self.use_layernorm:
            out = self.norm(out)
        return self.fc(out)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, num_channels=[64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = input_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.transpose(1, 2)
        out = self.network(x)
        # out: [batch, channels, seq]
        out = out[:, :, -1]  # last time step
        return self.fc(out)


