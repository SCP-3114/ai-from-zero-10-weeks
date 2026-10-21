"""Week 1 example: command-line arguments and logging."""

import argparse
import logging
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a short training process")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--log-file", type=Path, default=Path("training.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be greater than zero")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    for epoch in range(1, args.epochs + 1):
        logging.info("epoch=%d/%d loss=%.4f", epoch, args.epochs, 1.0 / epoch)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
