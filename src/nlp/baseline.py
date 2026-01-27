from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .run_utils import ensure_dirs, load_json, save_json


class BaselineTrainer:
    def __init__(
        self,
        max_features: int = 50000,
        ngram_min: int = 1,
        ngram_max: int = 2,
        C: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.C = C
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
        )
        self.model = LogisticRegression(
            max_iter=2000,
            C=C,
            random_state=seed,
        )

    def fit(self, train_texts: List[str], train_labels: List[int]) -> None:
        features = self.vectorizer.fit_transform(train_texts)
        self.model.fit(features, train_labels)

    def predict(self, texts: List[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        return self.model.predict(features)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        return self.model.predict_proba(features)

    def save(self, model_dir: Path, meta: Dict[str, object]) -> None:
        baseline_dir = model_dir / "baseline"
        ensure_dirs(baseline_dir)
        joblib.dump(self.model, baseline_dir / "model.joblib")
        joblib.dump(self.vectorizer, baseline_dir / "vectorizer.joblib")
        save_json(baseline_dir / "meta.json", meta)


class BaselinePredictor:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        model: LogisticRegression,
        id2label: Dict[str, str],
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.id2label = id2label

    @classmethod
    def load(cls, model_dir: Path) -> "BaselinePredictor":
        vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        model = joblib.load(model_dir / "model.joblib")
        meta = load_json(model_dir / "meta.json")
        id2label = meta.get("id2label", {"0": "neg", "1": "pos"})
        return cls(vectorizer, model, id2label)

    def predict(self, texts: List[str]) -> List[Dict[str, object]]:
        features = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(features)
        pred_ids = np.argmax(probs, axis=1)

        results = []
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
