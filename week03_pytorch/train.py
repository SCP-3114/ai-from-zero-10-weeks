import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from week03_pytorch.data import create_data_loaders
from week03_pytorch.model import FashionMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练Fashion-MNIST MLP"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/week03"),
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 提高复现性，可能略微降低速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清除上一个batch留下的梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(images)
        loss = loss_function(logits, labels)

        # 自动计算梯度
        loss.backward()

        # 根据梯度更新参数
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def run_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 不建立计算图，减少内存使用
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "config": config,
    }

    torch.save(checkpoint, path)


def save_curves(
    output_path: Path,
    history: dict[str, list[float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        history["train_loss"],
        label="train",
    )
    axes[0].plot(
        history["validation_loss"],
        label="validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(
        history["train_accuracy"],
        label="train",
    )
    axes[1].plot(
        history["validation_accuracy"],
        label="validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs必须大于0")

    if args.learning_rate <= 0:
        raise ValueError("learning-rate必须大于0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    device = select_device()

    print(f"使用设备：{device}")

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    config_path = args.output_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    data_loaders = create_data_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = FashionMLP(
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = -1.0
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_training_epoch(
            model,
            data_loaders.train,
            loss_function,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = run_evaluation(
            model,
            data_loaders.validation,
            loss_function,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"epoch={epoch:02d}/{args.epochs}, "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_loss,
                validation_accuracy,
                config,
            )

            print("  已保存新的最佳模型")

    history_path = args.output_dir / "history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    save_curves(
        args.output_dir / "training_curves.png",
        history,
    )

    print("\n训练结束")
    print(f"最佳验证准确率：{best_validation_accuracy:.4f}")
    print(f"最佳模型：{checkpoint_path}")


if __name__ == "__main__":
    main()