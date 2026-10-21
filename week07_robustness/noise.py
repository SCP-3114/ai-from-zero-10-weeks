"""Measure accuracy under reproducible Gaussian image noise."""

import argparse
import csv
from pathlib import Path

import torch

from src.data import create_data_loaders
from src.model import FashionMLP
from src.utils import select_device


def add_noise(images: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return (images + torch.randn(images.shape, generator=generator) * sigma).clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(sigma < 0 for sigma in args.sigmas):
        parser.error("all sigmas must be nonnegative")
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
    rows = []
    for sigma in args.sigmas:
        generator = torch.Generator().manual_seed(config["seed"])
        correct = total = 0
        with torch.inference_mode():
            for images, labels in loader:
                noisy = add_noise(images, sigma, generator).to(device)
                predictions = model(noisy).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += labels.numel()
        rows.append({"sigma": sigma, "accuracy": correct / total, "samples": total})
        print(f"sigma={sigma:g} accuracy={correct / total:.4f}")
    output = args.output or args.checkpoint.parent / "week07_noise.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sigma", "accuracy", "samples"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
