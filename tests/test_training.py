import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import FashionMLP
from src.train import run_evaluation, run_training_epoch


def test_one_training_epoch_runs_without_downloading_data() -> None:
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(32, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (32,), generator=generator)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = FashionMLP(hidden_size=16, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loss, train_accuracy = run_training_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )
    evaluation_loss, evaluation_accuracy = run_evaluation(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )

    assert train_loss > 0 and evaluation_loss > 0
    assert 0 <= train_accuracy <= 1
    assert 0 <= evaluation_accuracy <= 1
