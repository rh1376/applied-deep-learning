from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    source_path: str
    text: str
    metadata: dict[str, Any]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _make_doc_id(path: Path) -> str:
    resolved = str(path.resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()


def _build_metadata(path: Path) -> dict[str, Any]:
    modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "modified_time": modified,
    }


def _load_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires PyMuPDF. Install it with: pip install pymupdf"
        ) from exc

    pages: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def list_files(raw_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist or is not a directory: {raw_dir}")

    normalized_exts = {ext.lower() for ext in exts}
    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    return sorted(files)


def load_document(path: Path) -> Document:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document path does not exist or is not a file: {path}")

    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".pdf":
        text = _load_pdf_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    cleaned_text = _clean_text(text)
    return Document(
        doc_id=_make_doc_id(path),
        source_path=str(path.resolve()),
        text=cleaned_text,
        metadata=_build_metadata(path),
    )


def load_corpus(raw_dir: Path) -> list[Document]:
    supported_exts = (".txt", ".md", ".pdf")
    files = list_files(raw_dir=raw_dir, exts=supported_exts)
    return [load_document(path) for path in files]