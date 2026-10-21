"""Evaluate a Fashion-MNIST checkpoint with a confusion matrix."""

import argparse
import csv
from pathlib import Path

import torch

from src.data import create_data_loaders
from src.evaluate import CLASS_NAMES
from src.model import FashionMLP
from src.utils import save_json, select_device


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor, classes: int = 10) -> torch.Tensor:
    """Rows are true labels; columns are predictions."""
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be matching 1-D tensors")
    if torch.any((labels < 0) | (labels >= classes) | (predictions < 0) | (predictions >= classes)):
        raise ValueError("class index out of range")
    return torch.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = FashionMLP(config["hidden_size"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = create_data_loaders(
        config["data_directory"], batch_size=config["batch_size"],
        validation_ratio=config["validation_ratio"], seed=config["seed"],
        num_workers=config["num_workers"],
    ).test
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            matrix += confusion_matrix(labels, predictions)
    output = args.output or args.checkpoint.parent / "week06_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        support = int(matrix[index].sum())
        correct = int(matrix[index, index])
        rows.append({"class": name, "support": support, "correct": correct,
                     "accuracy": correct / support if support else None})
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "support", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    save_json({"class_names": CLASS_NAMES, "matrix": matrix.tolist()}, output / "confusion_matrix.json")
    print(f"overall_accuracy={matrix.diag().sum().item() / matrix.sum().item():.4f}")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
