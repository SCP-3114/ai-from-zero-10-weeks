"""Predict a local clothing image with the saved Fashion-MNIST model."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.evaluate import CLASS_NAMES
from src.model import FashionMLP
from src.utils import select_device


def prepare_image(path: Path, invert: bool = False) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        if invert:
            image = ImageOps.invert(image)
        image = ImageOps.pad(image, (28, 28), color=0)
        return transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])(image).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--invert", action="store_true", help="Invert dark clothing on light background")
    args = parser.parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = FashionMLP(config["hidden_size"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = model(prepare_image(args.image, args.invert).to(device)).softmax(dim=1)[0]
    for index in probabilities.argsort(descending=True)[:3]:
        print(f"{CLASS_NAMES[index]}: {probabilities[index].item():.3f}")


if __name__ == "__main__":
    main()
