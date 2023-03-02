from torch import nn as nn


class DecReadout(nn.Module):
    """
    Readout layer for decoder.
    """

    def __init__(self, feature_dim, node_num, hidden_dim=256, feature_reduction_dim=8):
        super().__init__()

        self.dim_reduction = nn.Sequential(
            nn.Linear(feature_dim, feature_reduction_dim),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(feature_reduction_dim * node_num, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 8, 2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dim_reduction(x)
        x = x.reshape((batch_size, -1))
        return self.fc(x)
