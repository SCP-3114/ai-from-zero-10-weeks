from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_data_loaders(
    data_directory: str = "data",
    batch_size: int = 128,
    validation_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaders:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_training_dataset = datasets.FashionMNIST(
        root=data_directory,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_directory,
        train=False,
        download=True,
        transform=transform,
    )

    validation_size = int(
        len(full_training_dataset) * validation_ratio
    )
    training_size = (
        len(full_training_dataset) - validation_size
    )

    split_generator = torch.Generator().manual_seed(seed)

    training_dataset, validation_dataset = random_split(
        full_training_dataset,
        [training_size, validation_size],
        generator=split_generator,
    )

    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=shuffle_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )