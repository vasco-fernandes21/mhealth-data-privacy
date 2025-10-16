import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMWESAD(nn.Module):
    """
    PyTorch CNN-LSTM model for WESAD stress detection.

    Matches the TensorFlow baseline structure:
    - Conv1D(32, k=7) + BN + MaxPool(4) + Dropout
    - Conv1D(64, k=5) + BN + MaxPool(2) + Dropout
    - LSTM(32)
    - Dense(32) + Dropout
    - Dense(2)
    """

    def __init__(
        self,
        input_channels: int = 14,
        num_classes: int = 2,
        dropout_conv: float = 0.2,
        dropout_dense: float = 0.4,
    ) -> None:
        super().__init__()

        # Match TF: Conv(32) -> BN -> MaxPool(4) -> Dropout(0.2)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.drop1 = nn.Dropout(p=0.2)

        # Match TF: Conv(64) -> BN -> MaxPool(2) -> Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(p=0.2)

        # After conv/pool blocks, feature dim is 64; sequence length is reduced by 8x
        # LSTM expects input (batch, seq, features)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # Dropout after LSTM to mirror TF (0.5)
        self.drop_lstm = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32, 32)
        self.drop3 = nn.Dropout(p=dropout_dense)  # 0.4 by default
        self.fc2 = nn.Linear(32, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape: (batch, channels, timesteps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Prepare for LSTM: (batch, seq, features)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop_lstm(x)

        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


