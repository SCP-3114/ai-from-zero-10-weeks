"""Fashion-MNIST data loading and deterministic train/validation splitting."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    data_directory: str | Path = "data",
    batch_size: int = 128,
    validation_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = True,
) -> DataLoaders:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    full_training_dataset = datasets.FashionMNIST(
        root=str(data_directory), train=True, download=download, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_directory), train=False, download=download, transform=transform
    )

    validation_size = int(len(full_training_dataset) * validation_ratio)
    training_size = len(full_training_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    training_dataset, validation_dataset = random_split(
        full_training_dataset,
        [training_size, validation_size],
        generator=split_generator,
    )

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return DataLoaders(
        train=DataLoader(
            training_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **common_options,
        ),
        validation=DataLoader(validation_dataset, shuffle=False, **common_options),
        test=DataLoader(test_dataset, shuffle=False, **common_options),
    )
