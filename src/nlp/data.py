from __future__ import annotations

import hashlib
from typing import Dict, List

from datasets import load_dataset


def _fingerprint(dataset_name: str, seed: int, val_size: float, counts: Dict[str, int]) -> str:
    payload = f"{dataset_name}|{seed}|{val_size}|{counts['train']}|{counts['val']}|{counts['test']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_imdb_splits(seed: int = 42, val_size: float = 0.1) -> Dict[str, object]:
    dataset = load_dataset("stanfordnlp/imdb")
    train_valid = dataset["train"].train_test_split(test_size=val_size, seed=seed)

    train_texts: List[str] = train_valid["train"]["text"]
    train_labels: List[int] = train_valid["train"]["label"]
    val_texts: List[str] = train_valid["test"]["text"]
    val_labels: List[int] = train_valid["test"]["label"]
    test_texts: List[str] = dataset["test"]["text"]
    test_labels: List[int] = dataset["test"]["label"]

    label2id = {"neg": 0, "pos": 1}
    id2label = {"0": "neg", "1": "pos"}

    counts = {
        "train": len(train_texts),
        "val": len(val_texts),
        "test": len(test_texts),
    }

    return {
        "train": {"texts": train_texts, "labels": train_labels},
        "val": {"texts": val_texts, "labels": val_labels},
        "test": {"texts": test_texts, "labels": test_labels},
        "label2id": label2id,
        "id2label": id2label,
        "fingerprint": _fingerprint("imdb", seed, val_size, counts),
    }
