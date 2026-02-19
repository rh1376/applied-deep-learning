from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def get_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    markers = ("pyproject.toml", ".git", "setup.py", "requirements.txt")

    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent

    try:
        return current_file.parents[2]
    except IndexError:
        return current_file.parent


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw else default


@dataclass(slots=True)
class HelpdeskRAGConfig:
    repo_root: Path = field(default_factory=get_repo_root)

    raw_dir: Path | str | None = None
    index_dir: Path | str | None = None

    chunk_size: int | None = None
    chunk_overlap: int | None = None
    top_k: int | None = None

    embedding_model_name: str | None = None
    llm_backend: str | None = None
    llm_model_name: str | None = None

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()

        raw_dir_value = self.raw_dir if self.raw_dir is not None else os.getenv("RAW_DIR", "data/helpdesk_kb/raw")
        index_dir_value = self.index_dir if self.index_dir is not None else os.getenv("INDEX_DIR", "data/helpdesk_kb/index")
        self.raw_dir = _resolve_path(raw_dir_value, self.repo_root)
        self.index_dir = _resolve_path(index_dir_value, self.repo_root)

        self.chunk_size = self.chunk_size if self.chunk_size is not None else _env_int("CHUNK_SIZE", 800)
        self.chunk_overlap = self.chunk_overlap if self.chunk_overlap is not None else _env_int("CHUNK_OVERLAP", 120)
        self.top_k = self.top_k if self.top_k is not None else _env_int("TOP_K", 5)

        self.embedding_model_name = self.embedding_model_name or _env_str(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_backend = self.llm_backend or _env_str("LLM_BACKEND", "transformers")
        self.llm_model_name = self.llm_model_name or _env_str("LLM_MODEL_NAME", "google/flan-t5-base")


def get_config() -> HelpdeskRAGConfig:
    return HelpdeskRAGConfig()