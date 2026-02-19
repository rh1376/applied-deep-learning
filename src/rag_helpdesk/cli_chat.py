from __future__ import annotations

import argparse
from pathlib import Path

from src.rag_helpdesk.config import get_config
from src.rag_helpdesk.ingest import load_rag_engine


def _parse_args() -> argparse.Namespace:
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Simple CLI chat for Helpdesk RAG")
    parser.add_argument("--index_dir", type=Path, default=cfg.index_dir, help="Path to FAISS index directory")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=cfg.embedding_model_name,
        help="Embedding model name",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=cfg.llm_model_name,
        help="LLM model name",
    )
    parser.add_argument("--top_k", type=int, default=cfg.top_k, help="Number of retrieved chunks")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    engine = load_rag_engine(
        index_dir=Path(args.index_dir),
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        top_k=args.top_k,
    )

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() == "quit":
            break

        result = engine.answer(query)

        print("Answer:")
        print(result.get("answer", ""))

        citations = result.get("citations", [])
        if citations:
            print("Citations:")
            for idx, item in enumerate(citations[: args.top_k], start=1):
                source_path = str(item.get("source_path", ""))
                score = float(item.get("score", 0.0))
                print(f"{idx}. {source_path} (score={score:.4f})")
        else:
            print("Citations: none")

        print()


if __name__ == "__main__":
    main()