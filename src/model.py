"""Neural-network models used by the project."""

import torch
from torch import nn


class FashionMLP(nn.Module):
    """A small multilayer perceptron for 28x28 grayscale images."""

    def __init__(self, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
