from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.rag_helpdesk.io_utils import Document


# ---------- Tunables (پیشنهادهای پیش‌فرض) ----------
# حداقل طول چانک (کاراکتر) تا breakpoint پاراگرافی را بپذیریم
MIN_CHUNK_CHARS = 200

# چانک‌های خیلی کوتاه را کلاً حذف می‌کنیم (برای جلوگیری از contextهای ناقص)
DROP_CHUNKS_SHORTER_THAN = 80

# برای snap کردن next_start به whitespace، اینقدر اطراف را می‌گردیم
SNAP_WINDOW = 40
# --------------------------------------------------


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
    # breakpoint = انتهای هر بلاک پاراگرافی (دو newline یا بیشتر)
    breaks = [match.end() for match in re.finditer(r"\n\s*\n+", text)]
    if not breaks or breaks[-1] != len(text):
        breaks.append(len(text))
    return breaks


def _snap_to_whitespace(text: str, pos: int, window: int) -> int:
    """
    تلاش می‌کند pos را به نزدیک‌ترین whitespace بعد از pos منتقل کند
    تا start چانک بعدی وسط کلمه نیفتد.
    """
    n = len(text)
    if pos <= 0:
        return 0
    if pos >= n:
        return n

    left = max(0, pos - window)
    right = min(n, pos + window)
    segment = text[left:right]
    rel = pos - left

    # اول: نزدیک‌ترین whitespace بعد از pos
    for j in range(rel, len(segment)):
        if segment[j].isspace():
            return min(n, left + j + 1)

    # اگر پیدا نشد: نزدیک‌ترین whitespace قبل از pos
    for j in range(rel, -1, -1):
        if segment[j].isspace():
            return min(n, left + j + 1)

    # اگر هیچ whitespace نبود همون pos
    return pos


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

        # breakpoint پاراگرافی را فقط وقتی قبول می‌کنیم که چانک خیلی کوتاه نشود
        preferred_end = None
        for bp in breakpoints:
            if bp <= start:
                continue
            if bp > max_end:
                break
            if (bp - start) >= MIN_CHUNK_CHARS:
                preferred_end = bp

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

        # جلوگیری از افتادن start وسط کلمه
        next_start = _snap_to_whitespace(text, next_start, SNAP_WINDOW)

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

        chunk_index = 0
        for _, (chunk_text_value, start_char, end_char) in enumerate(doc_spans):
            text_clean = chunk_text_value.strip()

            # حذف چانک‌های خیلی کوتاه (جلوگیری از contextهای ناقص مثل "ی بیمه تکمیلی")
            if len(text_clean) < DROP_CHUNKS_SHORTER_THAN:
                continue

            metadata = {
                "source_path": doc.source_path,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
            }

            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.doc_id, chunk_index, start_char, end_char),
                    doc_id=doc.doc_id,
                    text=text_clean,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    return chunks