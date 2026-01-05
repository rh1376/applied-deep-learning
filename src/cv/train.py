from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .data import CIFAR10_CLASSES, get_dataloaders
from .model import create_resnet18, get_trainable_parameters


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def predict_all(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="test", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    output_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(CIFAR10_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CIFAR10_CLASSES)),
        yticks=range(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title="CIFAR-10 Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    num_classes: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": "resnet18",
        "num_classes": num_classes,
    }
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["feature", "finetune"], default="feature")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = create_resnet18(num_classes=len(CIFAR10_CLASSES), mode=args.mode)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        get_trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    checkpoint_path = Path("models/cv/resnet18_cifar10.pt")

    history: Dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path, num_classes=len(CIFAR10_CLASSES))

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    labels, preds = predict_all(model, test_loader, device)
    save_confusion_matrix(labels, preds, Path("reports/cv/confusion_matrix.png"))

    metrics = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "last_epoch": history.get("epoch", 0),
        "last_train_loss": history.get("train_loss", 0.0),
        "last_val_loss": history.get("val_loss", 0.0),
    }
    metrics_path = Path("reports/cv/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
