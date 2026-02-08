from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.defect_cv.data import create_dataloaders
from src.defect_cv.model import DefectCNN_Base, DefectCNN_Small, count_params, get_model


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior (may reduce performance).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from logits and target labels."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(total) if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        total += batch_size

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = float(running_correct) / float(total) if total > 0 else 0.0
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """Evaluate model and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            total += batch_size

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = float(running_correct) / float(total) if total > 0 else 0.0
    return avg_loss, avg_acc


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Compute confusion matrix for a dataset without sklearn."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
            else:
                logits = model(images)

            preds = torch.argmax(logits, dim=1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                matrix[t.long(), p.long()] += 1

    return matrix


def save_confusion_matrix_txt(
    matrix: torch.Tensor,
    class_names: Iterable[str],
    output_path: Path,
) -> None:
    """Save confusion matrix to a human-readable text file."""
    names = list(class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    header = "true\\pred\t" + "\t".join(names)
    lines.append(header)

    for idx, name in enumerate(names):
        row_values = "\t".join(str(int(v)) for v in matrix[idx].tolist())
        lines.append(f"{name}\t{row_values}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _get_model(name: str, num_classes: int) -> nn.Module:
    try:
        return get_model(name=name, num_classes=num_classes)
    except Exception:
        if name == "small":
            return DefectCNN_Small(num_classes=num_classes)
        if name == "base":
            return DefectCNN_Base(num_classes=num_classes)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train defect classification model.")
    parser.add_argument("--data_dir", type=str, default="data/defect/neu_cls/raw")
    parser.add_argument("--model", type=str, choices=["small", "base"], default="base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out_dir", type=str, default="models/defect_cv")
    parser.add_argument("--report_dir", type=str, default="reports/defect_cv")

    args = parser.parse_args()

    set_seed(args.seed)

    device = _resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names, class_weights = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    num_classes = len(class_names)
    model = _get_model(args.model, num_classes=num_classes).to(device)

    try:
        num_params = count_params(model)
    except Exception:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        history.append(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), last_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    confusion = compute_confusion_matrix(
        model=model,
        loader=val_loader,
        num_classes=num_classes,
        device=device,
        use_amp=use_amp,
    )
    save_confusion_matrix_txt(confusion, class_names, report_dir / "confusion_matrix.txt")

    metrics = {
        "model_name": args.model,
        "num_params": num_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
    }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "Defect CV Training Summary",
        f"Model: {args.model}",
        f"Parameters: {num_params}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"LR: {args.lr}",
        f"Weight decay: {args.weight_decay}",
        f"Seed: {args.seed}",
        f"Device: {device}",
        f"AMP: {use_amp}",
        f"Best epoch: {best_epoch}",
        f"Best val acc: {best_val_acc:.4f}",
        f"Best checkpoint: {best_path}",
        f"Last checkpoint: {last_path}",
    ]
    (report_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
