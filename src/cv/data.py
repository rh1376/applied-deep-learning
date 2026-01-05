from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir.mkdir(parents=True, exist_ok=True)

    full_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=_train_transform(),
        download=True,
    )

    full_train_for_val = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=_eval_transform(),
        download=True,
    )

    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        range(len(full_train)), [train_len, val_len], generator=generator
    )

    train_dataset = Subset(full_train, train_subset.indices)
    val_dataset = Subset(full_train_for_val, val_subset.indices)

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=_eval_transform(),
        download=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
