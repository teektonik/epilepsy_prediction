# Models

import torch.nn.functional as F
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, input_size: int, hidden_nodes: int=128, lstm_layers: int=4):
        super(BaselineModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_nodes, lstm_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_nodes, 2)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.rnn(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.sigmoid(self.fc(lstm_out))
        
        return output
    