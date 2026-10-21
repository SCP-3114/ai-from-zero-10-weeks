"""CIFAR-10 loading with a fixed, leakage-free validation split."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    data_directory: str | Path,
    batch_size: int,
    validation_ratio: float,
    seed: int,
    augmentation: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = True,
) -> DataLoaders:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")

    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    training_steps: list[object] = []
    if augmentation:
        training_steps.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    training_steps.extend(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    training_transform = transforms.Compose(training_steps)

    training_source = datasets.CIFAR10(
        root=str(data_directory), train=True, download=download, transform=training_transform
    )
    validation_source = datasets.CIFAR10(
        root=str(data_directory), train=True, download=False, transform=evaluation_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_directory), train=False, download=download, transform=evaluation_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(training_source), generator=generator).tolist()
    validation_size = int(len(indices) * validation_ratio)
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    training_dataset = Subset(training_source, training_indices)
    validation_dataset = Subset(validation_source, validation_indices)

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return DataLoaders(
        train=DataLoader(
            training_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **options,
        ),
        validation=DataLoader(validation_dataset, shuffle=False, **options),
        test=DataLoader(test_dataset, shuffle=False, **options),
    )
