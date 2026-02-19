from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. Install it with: "
                "pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._resolve_dim()

    def _resolve_dim(self) -> int:
        dim = None
        get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            dim = get_dim()

        if dim is None:
            sample = self._model.encode(
                ["dimension probe"],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            sample = np.asarray(sample)
            if sample.ndim != 2 or sample.shape[1] <= 0:
                raise ValueError("Unable to infer embedding dimension from model output")
            dim = int(sample.shape[1])

        return int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> "np.ndarray":
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}")

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        arr = arr / norms

        return arr.astype(np.float32, copy=False)