# Models

import torch
import torch.nn a nn


class BaselineModel(nn.Module):
    """
    Nasseri, M., Pal Attia, T., Joseph, B. et al. 
    Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning. 
    Sci Rep 11, 21935 (2021). https://doi.org/10.1038/s41598-021-01449-2
    """
    def __init__(self, input_size: int, hidden_nodes: int=128, lstm_layers: int=4):
        super(TestModel, self).__init__()
        self.rnn = nn.LSTM(input_size, lstm_layers, hidden_nodes, dropout=0.2)
        self.fc = nn.Linear(hidden_nodes * lstm_layers, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.fc(self.rnn(x)))