import pytest

torch = pytest.importorskip("torch")

from src.model import FashionMLP


def test_model_output_shape() -> None:
    model = FashionMLP(hidden_size=64, dropout=0.1)
    outputs = model(torch.randn(8, 1, 28, 28))
    assert outputs.shape == (8, 10)
    assert torch.isfinite(outputs).all()


def test_backward_produces_finite_gradients() -> None:
    model = FashionMLP(hidden_size=32)
    outputs = model(torch.randn(4, 1, 28, 28))
    loss = torch.nn.functional.cross_entropy(outputs, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("hidden_size", [0, -1])
def test_invalid_hidden_size_is_rejected(hidden_size: int) -> None:
    with pytest.raises(ValueError):
        FashionMLP(hidden_size=hidden_size)
