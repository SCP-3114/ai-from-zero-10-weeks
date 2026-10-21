import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_evaluation, run_training_epoch
from week04_cnn.model import CIFARMLP, SimpleCNN, create_model
from week04_cnn.train import load_experiment_config, validate_controlled_experiments


@pytest.mark.parametrize("model", [CIFARMLP(), SimpleCNN()])
def test_cifar_model_output_shape(model) -> None:
    outputs = model(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, 10)
    assert torch.isfinite(outputs).all()


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("transformer")


def test_experiment_config_is_strictly_controlled() -> None:
    config = load_experiment_config(__import__("pathlib").Path("configs/cifar10_experiments.json"))
    validate_controlled_experiments(config["experiments"])


def test_two_changes_are_rejected() -> None:
    experiments = {
        "baseline": {"model": "cnn", "learning_rate": 0.001},
        "unfair": {"model": "mlp", "learning_rate": 0.01},
    }
    with pytest.raises(ValueError, match="exactly one"):
        validate_controlled_experiments(experiments)


def test_cnn_training_step_without_dataset_download() -> None:
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(8, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (8,), generator=generator)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = SimpleCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss, accuracy = run_training_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )
    evaluation_loss, evaluation_accuracy = run_evaluation(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert loss > 0 and evaluation_loss > 0
    assert 0 <= accuracy <= 1
    assert 0 <= evaluation_accuracy <= 1
