from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .baseline import BaselineTrainer
from .data import load_imdb_splits
from .metrics_utils import (
    compute_metrics,
    save_classification_report_txt,
    save_confusion_matrix_png,
    save_misclassified_csv,
)
from .run_utils import get_device, get_run_dirs, make_run_id, save_json, ensure_dirs
from .transformer import train_distilbert, build_dataset


def _label_names(id2label: Dict[str, str]) -> List[str]:
    indices = sorted(int(k) for k in id2label.keys())
    return [id2label[str(i)] for i in indices]


def _hardware_info() -> Dict[str, object]:
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_name"] = torch.cuda.get_device_name(0)
    return info


def _save_reports(
    report_dir: Path,
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    id2label: Dict[str, str],
) -> Dict[str, object]:
    labels = sorted(int(k) for k in id2label.keys())
    label_names = _label_names(id2label)
    metrics = compute_metrics(y_true, y_pred, y_proba, id2label)

    save_json(report_dir / "metrics.json", metrics)
    save_confusion_matrix_png(
        report_dir / "confusion_matrix.png",
        y_true,
        y_pred,
        labels,
        label_names,
    )
    save_classification_report_txt(
        report_dir / "classification_report.txt",
        y_true,
        y_pred,
        label_names,
    )
    y_score = y_proba.max(axis=1)
    save_misclassified_csv(
        report_dir / "misclassified.csv",
        texts,
        y_true,
        y_pred,
        y_score,
        id2label,
    )
    return metrics


def run_baseline(args: argparse.Namespace, data: Dict[str, object], run_id: str) -> Dict[str, object]:
    model_dir, report_dir = get_run_dirs(run_id)
    ensure_dirs(model_dir, report_dir)

    trainer = BaselineTrainer(
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        C=args.C,
        seed=args.seed,
    )
    trainer.fit(data["train"]["texts"], data["train"]["labels"])

    test_texts = data["test"]["texts"]
    y_true = np.asarray(data["test"]["labels"])
    y_pred = trainer.predict(test_texts)
    y_proba = trainer.predict_proba(test_texts)

    metrics = _save_reports(
        report_dir,
        test_texts,
        y_true,
        y_pred,
        y_proba,
        data["id2label"],
    )

    meta = {
        "run_id": run_id,
        "model_type": "baseline",
        "created_at": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "val_size": args.val_size,
        "dataset": "imdb",
        "fingerprint": data["fingerprint"],
        "label2id": data["label2id"],
        "id2label": data["id2label"],
        "hyperparams": {
            "max_features": args.max_features,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "C": args.C,
        },
        "hardware": _hardware_info(),
    }

    trainer.save(model_dir, meta)
    save_json(report_dir / "meta.json", meta)
    return {
        "run_id": run_id,
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
        "metrics": metrics,
    }


def run_distilbert(args: argparse.Namespace, data: Dict[str, object], run_id: str) -> Dict[str, object]:
    model_dir, report_dir = get_run_dirs(run_id)
    ensure_dirs(model_dir, report_dir)
    distilbert_dir = model_dir / "distilbert"
    ensure_dirs(distilbert_dir)

    trainer, tokenizer = train_distilbert(
        train_texts=data["train"]["texts"],
        train_labels=data["train"]["labels"],
        val_texts=data["val"]["texts"],
        val_labels=data["val"]["labels"],
        model_dir=distilbert_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    test_dataset = build_dataset(
        data["test"]["texts"],
        data["test"]["labels"],
        tokenizer,
        args.max_length,
    )
    test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    y_true = test_output.label_ids
    y_proba = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
    y_pred = np.argmax(y_proba, axis=1)

    metrics = _save_reports(
        report_dir,
        data["test"]["texts"],
        y_true,
        y_pred,
        y_proba,
        data["id2label"],
    )

    meta = {
        "run_id": run_id,
        "model_type": "distilbert",
        "created_at": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "val_size": args.val_size,
        "dataset": "imdb",
        "fingerprint": data["fingerprint"],
        "label2id": data["label2id"],
        "id2label": data["id2label"],
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        },
        "hardware": _hardware_info(),
    }

    save_json(distilbert_dir / "meta.json", meta)
    save_json(report_dir / "meta.json", meta)
    return {
        "run_id": run_id,
        "model_dir": str(model_dir),
        "report_dir": str(report_dir),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment models on IMDB.")
    parser.add_argument("--model", choices=["baseline", "distilbert"], default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = make_run_id(args.model)
    data = load_imdb_splits(seed=args.seed, val_size=args.val_size)
    if args.model == "baseline":
        summary = run_baseline(args, data, run_id)
    else:
        summary = run_distilbert(args, data, run_id)

    metrics = summary["metrics"]
    print("Training complete")
    print(f"run_id: {summary['run_id']}")
    print(f"model_dir: {summary['model_dir']}")
    print(f"report_dir: {summary['report_dir']}")
    print(
        "metrics: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
