from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def generate_data(
    sample_count: int = 200,
    seed: int = 42,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)

    # X的形状：(N, 1)
    x = rng.uniform(-2.0, 2.0, size=(sample_count, 1))

    # noise的形状：(N, 1)
    noise = rng.normal(0.0, 0.3, size=(sample_count, 1))

    # y的形状：(N, 1)
    y = 3.0 * x + 2.0 + noise

    return x, y


def predict(
    x: FloatArray,
    weight: FloatArray,
    bias: FloatArray,
) -> FloatArray:
    """前向传播：y_pred = XW + b。"""
    return x @ weight + bias


def mean_squared_error(
    prediction: FloatArray,
    target: FloatArray,
) -> float:
    error = prediction - target
    return float(np.mean(error**2))


def train(
    x: FloatArray,
    y: FloatArray,
    epochs: int = 300,
    learning_rate: float = 0.05,
) -> tuple[FloatArray, FloatArray, list[float]]:
    rng = np.random.default_rng(0)

    # weight形状：(1, 1)
    weight = rng.normal(size=(1, 1))

    # bias形状：(1,)，会通过广播作用于所有样本
    bias = np.zeros(1)

    losses: list[float] = []
    sample_count = x.shape[0]

    for epoch in range(epochs):
        # ---------- 前向传播 ----------
        prediction = predict(x, weight, bias)
        error = prediction - y
        loss = mean_squared_error(prediction, y)
        losses.append(loss)

        # ---------- 反向传播 ----------
        # loss = mean((prediction - y)²)
        #
        # d_prediction = 2 * (prediction - y) / N
        gradient_prediction = 2.0 * error / sample_count

        # prediction = XW + b
        gradient_weight = x.T @ gradient_prediction
        gradient_bias = np.sum(
            gradient_prediction,
            axis=0,
        )

        # ---------- SGD更新 ----------
        weight -= learning_rate * gradient_weight
        bias -= learning_rate * gradient_bias

        if epoch % 50 == 0:
            print(
                f"epoch={epoch:3d}, "
                f"loss={loss:.6f}, "
                f"weight={weight.item():.4f}, "
                f"bias={bias.item():.4f}"
            )

    return weight, bias, losses


def save_plot(
    x: FloatArray,
    y: FloatArray,
    weight: FloatArray,
    bias: FloatArray,
    losses: list[float],
) -> None:
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(x[:, 0], y[:, 0], s=12, label="data")

    sorted_indices = np.argsort(x[:, 0])
    sorted_x = x[sorted_indices]
    sorted_prediction = predict(sorted_x, weight, bias)

    axes[0].plot(
        sorted_x[:, 0],
        sorted_prediction[:, 0],
        color="red",
        label="prediction",
    )
    axes[0].set_title("Linear regression")
    axes[0].legend()

    axes[1].plot(losses)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")

    figure.tight_layout()
    figure.savefig(
        output_directory / "linear_regression.png",
        dpi=150,
    )
    plt.close(figure)


def main() -> None:
    x, y = generate_data()

    weight, bias, losses = train(
        x,
        y,
        epochs=300,
        learning_rate=0.05,
    )

    print("\n训练完成")
    print(f"学习到的weight：{weight.item():.4f}")
    print(f"学习到的bias：{bias.item():.4f}")

    save_plot(x, y, weight, bias, losses)


if __name__ == "__main__":
    main()
