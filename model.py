import torch
import torch.nn as nn

import lightning as L
import torchmetrics


class LSTMDetector(L.LightningModule):
    def __init__(self, input_size: int, hidden_nodes: int = 128, lstm_layers: int = 4):
        """
        Initializes the LSTMDetector with the given parameters.

        Parameters:
        - input_size (int): Size of input features.
        - hidden_nodes (int): Number of nodes in the hidden layer.
        - lstm_layers (int): Number of LSTM layers.
        """
        super(LSTMDetector, self).__init__()

        self.rnn = nn.LSTM(
            input_size, hidden_nodes, lstm_layers, dropout=0.2, batch_first=True
        )
        self.fc = nn.Linear(hidden_nodes, 512)
        self.fc2 = nn.Linear(512, 2)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.criterion = nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1Score(task="binary", num_classes=2)
        self.recall = torchmetrics.Recall(task="binary", num_classes=2)
<<<<<<< HEAD
        self.precision = torchmetrics.classification.Precision(task='binary', num_classes=2)
        
    def forward(self, x):
=======
        self.precision = torchmetrics.classification.Precision(
            task="binary", num_classes=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
>>>>>>> main
        lstm_out, _ = self.rnn(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc2(self.relu(self.fc(lstm_out)))

        return output

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns:
        - torch.optim.Optimizer: Optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def evaluate(self, batch: tuple, stage: str = None) -> torch.Tensor:
        """
        Evaluate the model performance.

        Parameters:
        - batch (tuple): Input batch.
        - stage (str): Stage of evaluation (train/validation/test).

        Returns:
        - torch.Tensor: Loss value.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        f1 = self.f1_score(preds, targets)
        recall = self.recall(preds, targets)
        precision = self.precision(preds, targets)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True)
            self.log(f"{stage}_recall", recall, prog_bar=True)
            self.log(f"{stage}_precision", precision, prog_bar=True)

        if stage == "train":
            return loss

    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters:
        - train_batch (tuple): Input batch.
        - batch_idx (int): Batch index.

        Returns:
        - torch.Tensor: Loss value.
        """
        loss = self.evaluate(train_batch, "train")
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Validation step.

        Parameters:
        - batch (tuple): Input batch.
        - batch_idx (int): Batch index.
        """
        self.evaluate(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int):
        """
        Test step.

        Parameters:
        - batch (tuple): Input batch.
        - batch_idx (int): Batch index.
        """
        self.evaluate(batch, "test")
