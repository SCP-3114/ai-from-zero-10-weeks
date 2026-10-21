"""Train a Fashion-MNIST MLP and save a reproducible experiment."""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data import create_data_loaders
from src.model import FashionMLP
from src.utils import (
    count_trainable_parameters,
    create_logger,
    load_config,
    memory_report,
    save_json,
    select_device,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/fashion_mnist.json"))
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override config learning rate"
    )
    parser.add_argument("--run-name", default=None, help="Optional result directory name")
    return parser.parse_args()


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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


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
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "config": config,
        },
        path,
    )


def save_curves(path: Path, history: dict[str, list[float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train")
    axes[1].plot(history["validation_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if config["epochs"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("epochs and learning_rate must be greater than zero")

    set_random_seed(config["seed"])
    device = select_device()
    run_name = args.run_name or (
        f"{config['experiment_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    result_directory = Path(config["results_directory"]) / run_name
    result_directory.mkdir(parents=True, exist_ok=False)
    logger = create_logger(result_directory / "training.log")
    config["device"] = str(device)
    save_json(config, result_directory / "config.json")

    loaders = create_data_loaders(
        data_directory=config["data_directory"],
        batch_size=config["batch_size"],
        validation_ratio=config["validation_ratio"],
        seed=config["seed"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = FashionMLP(config["hidden_size"], config["dropout"]).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    parameter_count = count_trainable_parameters(model)
    logger.info("device=%s | trainable_parameters=%d", device, parameter_count)
    logger.info("config=%s", config)

    history = {key: [] for key in (
        "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"
    )}
    best_accuracy = -1.0
    start_time = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_training_epoch(
            model, loaders.train, loss_function, optimizer, device
        )
        validation_loss, validation_accuracy = run_evaluation(
            model, loaders.validation, loss_function, device
        )
        for key, value in (
            ("train_loss", train_loss), ("train_accuracy", train_accuracy),
            ("validation_loss", validation_loss),
            ("validation_accuracy", validation_accuracy),
        ):
            history[key].append(value)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(
            "epoch=%d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f time=%.2fs",
            epoch, config["epochs"], train_loss, train_accuracy,
            validation_loss, validation_accuracy, time.perf_counter() - epoch_start,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                result_directory / "best_model.pt", model, optimizer, epoch,
                validation_loss, validation_accuracy, config,
            )

    elapsed = time.perf_counter() - start_time
    save_json(history, result_directory / "history.json")
    save_curves(result_directory / "training_curves.png", history)
    save_json(
        {
            "best_validation_accuracy": best_accuracy,
            "training_seconds": elapsed,
            "trainable_parameters": parameter_count,
            "memory": memory_report(device),
        },
        result_directory / "summary.json",
    )
    logger.info("finished in %.2fs | best_val_acc=%.4f", elapsed, best_accuracy)
    logger.info("results=%s", result_directory.resolve())


if __name__ == "__main__":
    main()
