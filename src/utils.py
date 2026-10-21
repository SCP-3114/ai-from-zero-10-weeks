"""Shared configuration, logging, reproducibility, and reporting helpers."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)

    required = {
        "experiment_name", "data_directory", "results_directory", "seed",
        "epochs", "batch_size", "learning_rate", "hidden_size", "dropout",
        "validation_ratio", "num_workers",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    return config


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def create_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("summer_ai_foundation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def memory_report(device: torch.device) -> dict[str, float | str]:
    report: dict[str, float | str] = {"device": str(device)}
    if device.type == "cuda":
        report.update(
            allocated_mb=torch.cuda.memory_allocated(device) / 1024**2,
            reserved_mb=torch.cuda.memory_reserved(device) / 1024**2,
            peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
        )
    return report
