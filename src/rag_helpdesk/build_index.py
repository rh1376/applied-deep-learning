from __future__ import annotations

import argparse
from pathlib import Path

from src.rag_helpdesk.chunking import chunk_documents
from src.rag_helpdesk.config import get_config
from src.rag_helpdesk.embeddings import Embedder
from src.rag_helpdesk.io_utils import load_corpus
from src.rag_helpdesk.vector_store import FaissStore


def _parse_args() -> argparse.Namespace:
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Build FAISS index for Helpdesk RAG corpus")
    parser.add_argument("--raw_dir", type=Path, default=cfg.raw_dir, help="Path to raw documents directory")
    parser.add_argument("--index_dir", type=Path, default=cfg.index_dir, help="Path to output index directory")
    parser.add_argument("--chunk_size", type=int, default=cfg.chunk_size, help="Chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=cfg.chunk_overlap, help="Chunk overlap in characters")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=cfg.embedding_model_name,
        help="Sentence-transformers model name",
    )
    parser.add_argument("--device", type=str, default=None, help="Embedding device, e.g. cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()

    docs = load_corpus(raw_dir)
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)

    if not chunks:
        raise RuntimeError("No chunks were produced from the provided corpus")

    embedder = Embedder(model_name=args.embedding_model, device=args.device)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks], batch_size=args.batch_size)

    store = FaissStore(index_dir=index_dir)
    store.build(vectors=vectors, chunks=chunks)
    store.save()

    print(f"docs={len(docs)}")
    print(f"chunks={len(chunks)}")
    print(f"embedding_dim={embedder.dim}")
    print(f"index_file={store.index_path}")
    print(f"metadata_file={store.metadata_path}")


if __name__ == "__main__":
    main()