"""Run one controlled CIFAR-10 experiment and record its results."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.train import run_evaluation, run_training_epoch
from src.utils import count_trainable_parameters, create_logger, save_json, select_device, set_random_seed
from week04_cnn.data import create_data_loaders
from week04_cnn.model import create_model


EXPERIMENT_FIELDS = (
    "experiment", "model", "optimizer", "learning_rate", "batch_size",
    "weight_decay", "augmentation", "best_validation_accuracy",
    "test_accuracy", "training_seconds", "trainable_parameters",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled CIFAR-10 experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/cifar10_experiments.json"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--experiment", help="Name of one experiment")
    selection.add_argument("--all", action="store_true", help="Run every configured experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for a smoke test")
    return parser.parse_args()


def load_experiment_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    if "shared" not in config or "experiments" not in config:
        raise ValueError("Config must contain shared and experiments sections")
    validate_controlled_experiments(config["experiments"])
    return config


def validate_controlled_experiments(experiments: dict[str, dict[str, Any]]) -> None:
    """Ensure every non-baseline experiment changes exactly one controlled variable."""
    if "baseline" not in experiments:
        raise ValueError("A baseline experiment is required")
    baseline = experiments["baseline"]
    for name, experiment in experiments.items():
        if name == "baseline":
            continue
        changed = [key for key in baseline if experiment.get(key) != baseline[key]]
        if len(changed) != 1:
            raise ValueError(
                f"{name} must change exactly one baseline variable; changed={changed}"
            )


def create_optimizer(
    name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name}")


def save_curves(path: Path, history: dict[str, list[float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split in ("train", "validation"):
        axes[0].plot(history[f"{split}_loss"], label=split)
        axes[1].plot(history[f"{split}_accuracy"], label=split)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="CIFAR-10 loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="CIFAR-10 accuracy")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result[field] for field in EXPERIMENT_FIELDS})


def run_experiment(
    experiment_name: str,
    shared: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    set_random_seed(shared["seed"])
    device = select_device()
    result_root = Path(shared["results_directory"])
    run_directory = result_root / (
        f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    logger = create_logger(run_directory / "training.log")
    full_config = {**shared, **experiment, "experiment": experiment_name, "device": str(device)}
    save_json(full_config, run_directory / "config.json")

    loaders = create_data_loaders(
        data_directory=shared["data_directory"],
        batch_size=experiment["batch_size"],
        validation_ratio=shared["validation_ratio"],
        seed=shared["seed"],
        augmentation=experiment["augmentation"],
        num_workers=shared["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = create_model(experiment["model"]).to(device)
    optimizer = create_optimizer(
        experiment["optimizer"], model, experiment["learning_rate"], experiment["weight_decay"]
    )
    loss_function = nn.CrossEntropyLoss()
    parameter_count = count_trainable_parameters(model)
    logger.info("experiment=%s device=%s parameters=%d", experiment_name, device, parameter_count)
    logger.info("controlled_config=%s", experiment)

    history = {key: [] for key in (
        "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"
    )}
    best_accuracy = -1.0
    checkpoint_path = run_directory / "best_model.pt"
    start = time.perf_counter()
    for epoch in range(1, shared["epochs"] + 1):
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
        logger.info(
            "epoch=%d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch, shared["epochs"], train_loss, train_accuracy,
            validation_loss, validation_accuracy,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": full_config},
                checkpoint_path,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_evaluation(model, loaders.test, loss_function, device)
    result = {
        "experiment": experiment_name,
        **experiment,
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_seconds": training_seconds,
        "trainable_parameters": parameter_count,
        "result_directory": str(run_directory),
    }
    save_json(history, run_directory / "history.json")
    save_json(result, run_directory / "summary.json")
    save_curves(run_directory / "training_curves.png", history)
    append_result(result_root / "experiments.csv", result)
    logger.info(
        "finished best_val_acc=%.4f test_acc=%.4f time=%.2fs",
        best_accuracy, test_accuracy, training_seconds,
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    shared = config["shared"].copy()
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        shared["epochs"] = args.epochs

    names = list(config["experiments"]) if args.all else [args.experiment]
    for name in names:
        if name not in config["experiments"]:
            raise KeyError(f"Unknown experiment: {name}")
        run_experiment(name, shared, config["experiments"][name])


if __name__ == "__main__":
    main()
