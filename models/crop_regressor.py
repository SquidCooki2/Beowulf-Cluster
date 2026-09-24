import torch
import torch.nn as nn

class CropYieldRegressor(nn.Module):
    """
    Multi-factor tabular regression model predicting agricultural harvest yield (metric tons/hectare)
    from soil temperature, rainfall, nitrogen, phosphorus, potassium, and sunlight hours.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Continuous yield output
        )

    def forward(self, x):
        return self.net(x)
