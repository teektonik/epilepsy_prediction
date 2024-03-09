# Models

import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

import lightning as L
import torchmetrics


class BaselineModel(L.LightningModule):
    """
    Nasseri, M., Pal Attia, T., Joseph, B. et al. 
    Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning. 
    Sci Rep 11, 21935 (2021). https://doi.org/10.1038/s41598-021-01449-2
    """
    def __init__(self,
                 input_size: int,
                 hidden_nodes: int=128, 
                 lstm_layers: int=4):
        
        super(BaselineModel, self).__init__()
      
        self.rnn = nn.LSTM(input_size, hidden_nodes, lstm_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_nodes, 512)
        self.fc2 = nn.Linear(512, 2)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.classification.Accuracy(task='binary', num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="binary", num_classes=2)
        self.recall = torchmetrics.Recall(task="binary", num_classes=2)
        self.precision = torchmetrics.classification.Precision(task='binary', num_classes=2)
            
    def forward(self, x):
        lstm_out, _ = self.rnn(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc2(self.relu(self.fc(lstm_out)))
        
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def evaluate(self, batch, stage=None):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        
        acc = self.acc_metric(preds, targets)
        f1 = self.f1_score(preds, targets)
        recall = self.recall(preds, targets)
        precison = self.precision(preds, targets)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True)
            self.log(f"{stage}_recall", recall, prog_bar=True)
            self.log(f"{stage}_precision", precison, prog_bar=True)
            
        if stage == 'train':
            return loss
            
    def training_step(self, train_batch, batch_idx):
        loss = self.evaluate(train_batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
        
        
    