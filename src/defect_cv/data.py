from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(train: bool) -> transforms.Compose:
    """
    Build torchvision transforms for training or validation.

    Training uses stronger augmentations; validation uses only resizing and normalization.
    Images are converted from grayscale to 3-channel tensors.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _compute_class_weights(targets: Iterable[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from target indices.
    """
    target_tensor = torch.tensor(list(targets), dtype=torch.long)
    if target_tensor.numel() == 0:
        raise ValueError("Cannot compute class weights from an empty target list.")
    class_counts = torch.bincount(target_tensor, minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    return weights


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    Create train/validation dataloaders from an ImageFolder dataset.

    Args:
        data_dir: Root directory of the ImageFolder dataset.
        batch_size: Batch size for loaders.
        val_split: Fraction of data reserved for validation.
        seed: RNG seed for reproducible split.
        num_workers: DataLoader worker processes.

    Returns:
        train_loader, val_loader, class_names, class_weights
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1 (exclusive).")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_path}")

    full_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(train=True),
    )
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in: {data_path}")

    class_names = full_dataset.classes

    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            "val_split results in an empty train or validation set. "
            "Adjust val_split or dataset size."
        )

    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Assign validation transforms without duplicating data on disk.
    val_subset.dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(train=False),
    )

    train_targets = [full_dataset.targets[i] for i in train_subset.indices]
    class_weights = _compute_class_weights(train_targets, len(class_names))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_subset, shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_subset, shuffle=False, drop_last=False, **loader_kwargs
    )

    return train_loader, val_loader, class_names, class_weights
