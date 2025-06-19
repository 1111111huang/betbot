import torch.nn as nn

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
