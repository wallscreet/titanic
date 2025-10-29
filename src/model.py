from torch import sigmoid
import torch.nn as nn


class TitanicNN(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 5
            nn.Linear(64, 32),
            nn.ReLU(),

            # Layer 6
            nn.Linear(32, 16),
            nn.Sigmoid(),
            
            # Output
            nn.Linear(16, 1),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        return self.net(x)