from __future__ import annotations

from pathlib import Path

from src.rag_helpdesk.embeddings import Embedder
from src.rag_helpdesk.rag import RAGEngine
from src.rag_helpdesk.retriever import Retriever
from src.rag_helpdesk.vector_store import FaissStore


def load_rag_engine(
    index_dir: Path,
    embedding_model_name: str,
    llm_model_name: str,
    top_k: int,
    device: str | None = None,
) -> RAGEngine:
    resolved_index_dir = Path(index_dir).expanduser().resolve()

    index_path = resolved_index_dir / "index.faiss"
    metadata_path = resolved_index_dir / "metadata.jsonl"

    if not resolved_index_dir.exists() or not resolved_index_dir.is_dir():
        raise FileNotFoundError(f"Index directory does not exist or is not a directory: {resolved_index_dir}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    embedder = Embedder(model_name=embedding_model_name, device=device)

    store = FaissStore(index_dir=resolved_index_dir)
    store.load()

    retriever = Retriever(embedder=embedder, store=store, top_k=top_k)
    rag_engine = RAGEngine(retriever=retriever, llm_model_name=llm_model_name)

    return rag_engine