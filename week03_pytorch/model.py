import torch
from torch import nn


class FashionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)