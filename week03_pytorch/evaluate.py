import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from week03_pytorch.data import create_data_loaders
from week03_pytorch.model import FashionMLP
from week03_pytorch.train import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试Fashion-MNIST模型"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/week03/best_model.pt"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{args.checkpoint}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["config"]

    model = FashionMLP(
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    data_loaders = create_data_loaders(
        batch_size=args.batch_size,
        seed=config["seed"],
        pin_memory=device.type == "cuda",
    )

    loss_function = nn.CrossEntropyLoss()

    test_loss, test_accuracy = run_evaluation(
        model,
        data_loaders.test,
        loss_function,
        device,
    )

    print(f"设备：{device}")
    print(f"模型来自第 {checkpoint['epoch']} 轮")
    print(
        "保存时验证准确率："
        f"{checkpoint['validation_accuracy']:.4f}"
    )
    print(f"测试loss：{test_loss:.4f}")
    print(f"测试accuracy：{test_accuracy:.4f}")


if __name__ == "__main__":
    main()