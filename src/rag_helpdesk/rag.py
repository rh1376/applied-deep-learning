from __future__ import annotations

from typing import Any

from src.rag_helpdesk.retriever import Retriever

DEFAULT_LLM_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 3500


def format_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""

    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        source_path = str(item.get("metadata", {}).get("source_path", ""))
        chunk_id = str(item.get("chunk_id", ""))
        text = str(item.get("text", "")).strip()
        blocks.append(f"[{idx}] source={source_path} chunk_id={chunk_id}\n{text}")

    return "\n\n".join(blocks)


class RAGEngine:
    def __init__(self, retriever: Retriever, llm_model_name: str, max_new_tokens: int = 256) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        model_name = llm_model_name or DEFAULT_LLM_MODEL

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for generation. Install it with: "
                "pip install transformers torch"
            ) from exc

        self.retriever = retriever
        self.llm_model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._generator = pipeline(
            task="text2text-generation",
            model=self.llm_model_name,
        )

    def _select_context_items(self, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        total_chars = 0

        for item in retrieved:
            text = str(item.get("text", ""))
            if not text:
                continue

            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break

            if len(text) > remaining:
                item_copy = dict(item)
                item_copy["text"] = text[:remaining].rstrip()
                if item_copy["text"]:
                    selected.append(item_copy)
                    total_chars += len(item_copy["text"])
                break

            selected.append(item)
            total_chars += len(text)

        return selected

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a helpdesk assistant. Answer ONLY using the provided context.\n"
            "If context is insufficient, reply exactly: \"I don't have enough information in the provided documents.\"\n"
            "Also include bullet citations that reference the context item numbers you used.\n\n"
            f"Question:\n{query.strip()}\n\n"
            f"Context:\n{context}\n\n"
            "Response format:\n"
            "Answer: <your grounded answer>\n"
            "Citations:\n"
            "- [n] brief note\n"
        )

    def answer(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        retrieved = self.retriever.retrieve(query)
        context_items = self._select_context_items(retrieved)
        context_texts = [str(item.get("text", "")) for item in context_items]

        citations = [
            {
                "score": float(item.get("score", 0.0)),
                "source_path": str(item.get("metadata", {}).get("source_path", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "doc_id": str(item.get("doc_id", "")),
            }
            for item in context_items
        ]

        if not context_items:
            return {
                "answer": "I don't have enough information in the provided documents.",
                "citations": citations,
                "context_used": context_texts,
            }

        prompt = self._build_prompt(query=query, context=format_context(context_items))
        outputs = self._generator(prompt, max_new_tokens=self.max_new_tokens)

        answer_text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                answer_text = str(first.get("generated_text", "")).strip()

        if not answer_text:
            answer_text = "I don't have enough information in the provided documents."

        return {
            "answer": answer_text,
            "citations": citations,
            "context_used": context_texts,
        }