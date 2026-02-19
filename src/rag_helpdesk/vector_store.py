from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.rag_helpdesk.chunking import Chunk


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "faiss is required for vector search. Install it with: pip install faiss-cpu"
        ) from exc
    return faiss


class FaissStore:
    def __init__(self, index_dir: Path) -> None:
        self._faiss = _import_faiss()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.jsonl"

        self.index: Any | None = None
        self._records: list[dict[str, Any]] = []

    def build(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be a 2D array, got shape {arr.shape}")
        if arr.shape[0] != len(chunks):
            raise ValueError(
                f"Number of vectors ({arr.shape[0]}) must match number of chunks ({len(chunks)})"
            )
        if arr.shape[0] == 0:
            raise ValueError("Cannot build FAISS index with zero vectors")

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dim = int(arr.shape[1])

        index = self._faiss.IndexFlatIP(dim)
        index.add(arr)

        self.index = index
        self._records = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Call build() or load() first.")

        self._faiss.write_index(self.index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.index = self._faiss.read_index(str(self.index_path))

        records: list[dict[str, Any]] = []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        self._records = records

        if self.index.ntotal != len(self._records):
            raise ValueError(
                "Index and metadata size mismatch: "
                f"index.ntotal={self.index.ntotal}, metadata={len(self._records)}"
            )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index is not loaded. Call build() or load() first.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not self._records:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        elif q.ndim == 2 and q.shape[0] == 1:
            pass
        else:
            raise ValueError(f"query_vec must be shape (d,) or (1, d), got {q.shape}")

        if q.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension mismatch: got {q.shape[1]}, expected {self.index.d}"
            )

        q = np.ascontiguousarray(q, dtype=np.float32)
        k = min(top_k, len(self._records))

        scores, indices = self.index.search(q, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = self._records[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": record["chunk_id"],
                    "doc_id": record["doc_id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                }
            )

        return results