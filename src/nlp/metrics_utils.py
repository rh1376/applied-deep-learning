from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)

from .run_utils import ensure_dirs, write_text


def _label_names(id2label: Dict[str, str]) -> List[str]:
    indices = sorted(int(k) for k in id2label.keys())
    return [id2label[str(i)] for i in indices]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    id2label: Dict[str, str],
) -> Dict[str, object]:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels = sorted(int(k) for k in id2label.keys())
    per_class = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    for idx, label_id in enumerate(labels):
        label_name = id2label[str(label_id)]
        per_class[label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }


def save_classification_report_txt(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    write_text(path, report)


def save_confusion_matrix_png(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    label_names: List[str],
) -> None:
    ensure_dirs(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_misclassified_csv(
    path: Path,
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    id2label: Dict[str, str],
    max_rows: int = 500,
) -> None:
    ensure_dirs(path.parent)
    rows = []
    for text, true_id, pred_id, score in zip(texts, y_true, y_pred, y_score):
        if int(true_id) == int(pred_id):
            continue
        rows.append(
            {
                "text": text,
                "true_label": id2label[str(int(true_id))],
                "pred_label": id2label[str(int(pred_id))],
                "score": float(score),
            }
        )

    df = pd.DataFrame(rows, columns=["text", "true_label", "pred_label", "score"])
    if not df.empty:
        df = df.head(max_rows)
    df.to_csv(path, index=False)
