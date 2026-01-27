from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_run_id(model_name: str, extra: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clean_model = model_name.strip().replace(" ", "_")
    if extra:
        clean_extra = extra.strip().replace(" ", "_")
        return f"{timestamp}-{clean_model}-{clean_extra}"
    return f"{timestamp}-{clean_model}"


def get_run_dirs(run_id: str) -> tuple[Path, Path]:
    root = _repo_root()
    model_dir = root / "models" / "nlp" / run_id
    report_dir = root / "reports" / "nlp" / run_id
    return model_dir, report_dir


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    ensure_dirs(path.parent)
    path.write_text(text, encoding="utf-8")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
