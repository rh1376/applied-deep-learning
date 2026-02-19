from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.rag_helpdesk.io_utils import Document


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _paragraph_breakpoints(text: str) -> list[int]:
    breaks = [match.end() for match in re.finditer(r"\n\s*\n+", text)]
    if not breaks or breaks[-1] != len(text):
        breaks.append(len(text))
    return breaks


def _chunk_text_with_offsets(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    _validate_chunk_params(chunk_size, overlap)

    if not text:
        return []

    n = len(text)
    breakpoints = _paragraph_breakpoints(text)
    spans: list[tuple[str, int, int]] = []
    start = 0

    while start < n:
        max_end = min(start + chunk_size, n)
        preferred_end = None
        for bp in breakpoints:
            if start < bp <= max_end:
                preferred_end = bp
            if bp > max_end:
                break

        end = preferred_end if preferred_end is not None else max_end
        if end <= start:
            end = min(start + chunk_size, n)
            if end <= start:
                break

        spans.append((text[start:end], start, end))

        if end >= n:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return spans


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    return [chunk for chunk, _, _ in _chunk_text_with_offsets(text, chunk_size, overlap)]


def _make_chunk_id(doc_id: str, chunk_index: int, start_char: int, end_char: int) -> str:
    payload = f"{doc_id}:{chunk_index}:{start_char}:{end_char}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []

    for doc in docs:
        doc_spans = _chunk_text_with_offsets(doc.text, chunk_size, overlap)
        for idx, (chunk_text_value, start_char, end_char) in enumerate(doc_spans):
            metadata = {
                "source_path": doc.source_path,
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
            }
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.doc_id, idx, start_char, end_char),
                    doc_id=doc.doc_id,
                    text=chunk_text_value,
                    metadata=metadata,
                )
            )

    return chunks