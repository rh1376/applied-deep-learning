from __future__ import annotations

from src.rag_helpdesk.embeddings import Embedder
from src.rag_helpdesk.vector_store import FaissStore


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissStore, top_k: int = 5) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = self.embedder.embed_texts([query], batch_size=1)[0]
        results = self.store.search(query_vector, top_k=k)

        normalized: list[dict] = []
        for item in results:
            normalized.append(
                {
                    "score": float(item["score"]),
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                }
            )

        return normalized