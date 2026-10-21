"""Evaluate a checkpoint and visualize incorrect predictions."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import create_data_loaders
from src.model import FashionMLP
from src.train import run_evaluation
from src.utils import save_json, select_device


CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Fashion-MNIST model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-errors", type=int, default=16)
    return parser.parse_args()


def save_incorrect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: Path,
    maximum_images: int = 16,
) -> int:
    model.eval()
    mistakes: list[tuple[torch.Tensor, int, int]] = []
    with torch.inference_mode():
        for images, labels in data_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for image, label, prediction in zip(images, labels, predictions):
                if label.item() != prediction.item():
                    mistakes.append((image, label.item(), prediction.item()))
                    if len(mistakes) >= maximum_images:
                        break
            if len(mistakes) >= maximum_images:
                break

    if not mistakes:
        return 0
    columns = 4
    rows = (len(mistakes) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image, label, prediction) in zip(axes.flat, mistakes):
        axis.imshow((image.squeeze(0) * 0.5 + 0.5).numpy(), cmap="gray")
        axis.set_title(f"true: {CLASS_NAMES[label]}\npred: {CLASS_NAMES[prediction]}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return len(mistakes)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = select_device()
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    config = checkpoint["config"]
    model = FashionMLP(config["hidden_size"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loaders = create_data_loaders(
        data_directory=config["data_directory"],
        batch_size=args.batch_size,
        validation_ratio=config["validation_ratio"],
        seed=config["seed"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    test_loss, test_accuracy = run_evaluation(
        model, loaders.test, nn.CrossEntropyLoss(), device
    )
    output_directory = args.checkpoint.parent
    mistake_count = save_incorrect_predictions(
        model, loaders.test, device, output_directory / "incorrect_predictions.png",
        args.max_errors,
    )
    metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy}
    save_json(metrics, output_directory / "test_metrics.json")
    print(f"device={device}")
    print(f"checkpoint_epoch={checkpoint['epoch']}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_accuracy:.4f}")
    print(f"visualized_errors={mistake_count}")


if __name__ == "__main__":
    main()
