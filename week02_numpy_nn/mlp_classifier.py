from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def generate_data(
    samples_per_class: int = 200,
    seed: int = 42,
) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [-2.0, -1.5],
            [2.0, -1.0],
            [0.0, 2.0],
        ]
    )

    feature_parts: list[FloatArray] = []
    label_parts: list[IntArray] = []

    for class_index, center in enumerate(centers):
        class_features = rng.normal(
            loc=center,
            scale=0.8,
            size=(samples_per_class, 2),
        )

        class_labels = np.full(
            samples_per_class,
            class_index,
            dtype=np.int64,
        )

        feature_parts.append(class_features)
        label_parts.append(class_labels)

    x = np.vstack(feature_parts)
    y = np.concatenate(label_parts)

    indices = rng.permutation(len(x))

    return x[indices], y[indices]


def split_data(
    x: FloatArray,
    y: IntArray,
) -> tuple[
    FloatArray,
    IntArray,
    FloatArray,
    IntArray,
    FloatArray,
    IntArray,
]:
    sample_count = len(x)

    train_end = int(sample_count * 0.70)
    validation_end = int(sample_count * 0.85)

    x_train = x[:train_end]
    y_train = y[:train_end]

    x_validation = x[train_end:validation_end]
    y_validation = y[train_end:validation_end]

    x_test = x[validation_end:]
    y_test = y[validation_end:]

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )


def standardize_data(
    x_train: FloatArray,
    x_validation: FloatArray,
    x_test: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    # 只能用训练集计算均值和标准差
    mean = np.mean(x_train, axis=0, keepdims=True)
    standard_deviation = np.std(
        x_train,
        axis=0,
        keepdims=True,
    )

    standard_deviation = np.where(
        standard_deviation == 0,
        1.0,
        standard_deviation,
    )

    return (
        (x_train - mean) / standard_deviation,
        (x_validation - mean) / standard_deviation,
        (x_test - mean) / standard_deviation,
    )


def relu(x: FloatArray) -> FloatArray:
    return np.maximum(0.0, x)


def softmax(logits: FloatArray) -> FloatArray:
    # 减去每行最大值，避免exp数值溢出
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True,
    )


def cross_entropy_loss(
    probabilities: FloatArray,
    labels: IntArray,
) -> float:
    sample_count = labels.shape[0]

    correct_class_probabilities = probabilities[
        np.arange(sample_count),
        labels,
    ]

    return float(
        -np.mean(
            np.log(correct_class_probabilities + 1e-12)
        )
    )


def accuracy(
    probabilities: FloatArray,
    labels: IntArray,
) -> float:
    predictions = np.argmax(probabilities, axis=1)
    return float(np.mean(predictions == labels))


class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)

        # He初始化，适合ReLU
        self.weight1 = rng.normal(
            0.0,
            np.sqrt(2.0 / input_size),
            size=(input_size, hidden_size),
        )
        self.bias1 = np.zeros((1, hidden_size))

        self.weight2 = rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_size),
            size=(hidden_size, output_size),
        )
        self.bias2 = np.zeros((1, output_size))

    def forward(
        self,
        x: FloatArray,
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        hidden_linear = x @ self.weight1 + self.bias1
        hidden_activation = relu(hidden_linear)
        logits = hidden_activation @ self.weight2 + self.bias2
        probabilities = softmax(logits)

        cache = {
            "x": x,
            "hidden_linear": hidden_linear,
            "hidden_activation": hidden_activation,
            "probabilities": probabilities,
        }

        return probabilities, cache

    def backward(
        self,
        labels: IntArray,
        cache: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        x = cache["x"]
        hidden_linear = cache["hidden_linear"]
        hidden_activation = cache["hidden_activation"]
        probabilities = cache["probabilities"]

        sample_count = labels.shape[0]

        # softmax + 交叉熵对logits的梯度
        gradient_logits = probabilities.copy()
        gradient_logits[np.arange(sample_count), labels] -= 1.0
        gradient_logits /= sample_count

        # 第二个线性层的梯度
        gradient_weight2 = (
            hidden_activation.T @ gradient_logits
        )
        gradient_bias2 = np.sum(
            gradient_logits,
            axis=0,
            keepdims=True,
        )

        # 梯度从第二层传回隐藏层
        gradient_hidden_activation = (
            gradient_logits @ self.weight2.T
        )

        # ReLU的梯度
        gradient_hidden_linear = (
            gradient_hidden_activation
            * (hidden_linear > 0)
        )

        # 第一个线性层的梯度
        gradient_weight1 = x.T @ gradient_hidden_linear
        gradient_bias1 = np.sum(
            gradient_hidden_linear,
            axis=0,
            keepdims=True,
        )

        return {
            "weight1": gradient_weight1,
            "bias1": gradient_bias1,
            "weight2": gradient_weight2,
            "bias2": gradient_bias2,
        }

    def update(
        self,
        gradients: dict[str, FloatArray],
        learning_rate: float,
    ) -> None:
        self.weight1 -= learning_rate * gradients["weight1"]
        self.bias1 -= learning_rate * gradients["bias1"]
        self.weight2 -= learning_rate * gradients["weight2"]
        self.bias2 -= learning_rate * gradients["bias2"]


def train(
    model: MLP,
    x_train: FloatArray,
    y_train: IntArray,
    x_validation: FloatArray,
    y_validation: IntArray,
    epochs: int = 500,
    learning_rate: float = 0.1,
) -> tuple[list[float], list[float]]:
    training_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(epochs):
        # 训练集前向传播
        training_probabilities, cache = model.forward(x_train)

        training_loss = cross_entropy_loss(
            training_probabilities,
            y_train,
        )

        # 只有训练集执行反向传播和参数更新
        gradients = model.backward(y_train, cache)
        model.update(gradients, learning_rate)

        # 验证集只进行前向传播
        validation_probabilities, _ = model.forward(
            x_validation
        )

        validation_loss = cross_entropy_loss(
            validation_probabilities,
            y_validation,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if epoch % 50 == 0:
            training_accuracy = accuracy(
                training_probabilities,
                y_train,
            )
            validation_accuracy = accuracy(
                validation_probabilities,
                y_validation,
            )

            print(
                f"epoch={epoch:3d}, "
                f"train_loss={training_loss:.4f}, "
                f"val_loss={validation_loss:.4f}, "
                f"train_acc={training_accuracy:.3f}, "
                f"val_acc={validation_accuracy:.3f}"
            )

    return training_losses, validation_losses


def save_loss_plot(
    training_losses: list[float],
    validation_losses: list[float],
) -> None:
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("NumPy MLP loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "numpy_mlp_loss.png",
        dpi=150,
    )
    plt.close()


def main() -> None:
    x, y = generate_data()

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = split_data(x, y)

    (
        x_train,
        x_validation,
        x_test,
    ) = standardize_data(
        x_train,
        x_validation,
        x_test,
    )

    model = MLP(
        input_size=2,
        hidden_size=16,
        output_size=3,
    )

    training_losses, validation_losses = train(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=500,
        learning_rate=0.1,
    )

    test_probabilities, _ = model.forward(x_test)
    test_loss = cross_entropy_loss(
        test_probabilities,
        y_test,
    )
    test_accuracy = accuracy(
        test_probabilities,
        y_test,
    )

    print("\n最终测试结果")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_accuracy:.3f}")

    save_loss_plot(
        training_losses,
        validation_losses,
    )


if __name__ == "__main__":
    main()
