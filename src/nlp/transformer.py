from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datetime import datetime

from .run_utils import ensure_dirs, get_device, load_json, save_json


def _build_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_length: int,
) -> Dataset:
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def build_dataset(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_length: int,
) -> Dataset:
    return _build_dataset(texts, labels, tokenizer, max_length)


def _trainer_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def train_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    model_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_length: int,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
) -> Tuple[Trainer, AutoTokenizer]:
    ensure_dirs(model_dir)
    model_name = "distilbert-base-uncased"
    id2label = {0: "neg", 1: "pos"}
    label2id = {"neg": 0, "pos": 1}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = _build_dataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = _build_dataset(val_texts, val_labels, tokenizer, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_trainer_metrics,
    )

    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_json(
        model_dir / "meta.json",
        {
            "model_type": "distilbert",
            "id2label": id2label,
            "label2id": label2id,
            "hyperparams": {
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "lr": lr,
                "max_length": max_length,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "seed": seed,
            },
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    return trainer, tokenizer


class TransformerPredictor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForSequenceClassification,
        id2label: Dict[str, str],
        device: torch.device,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.id2label = id2label
        self.device = device
        self.max_length = max_length

    @classmethod
    def load(cls, model_dir: Path) -> "TransformerPredictor":
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        meta = load_json(model_dir / "meta.json")
        id2label = meta.get("id2label", {"0": "neg", "1": "pos"})
        max_length = meta.get("hyperparams", {}).get("max_length", 256)
        device = get_device()
        return cls(tokenizer, model, id2label, device, max_length)

    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        self.model.eval()
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            pred_ids = np.argmax(probs, axis=1)
            for text_probs, pred_id in zip(probs, pred_ids):
                label = self.id2label[str(int(pred_id))]
                prob_map = {
                    self.id2label[str(idx)]: float(score)
                    for idx, score in enumerate(text_probs)
                }
                results.append(
                    {
                        "label": label,
                        "score": float(np.max(text_probs)),
                        "probs": prob_map,
                    }
                )
        return results

    def predict_one(self, text: str) -> Dict[str, object]:
        return self.predict([text])[0]
