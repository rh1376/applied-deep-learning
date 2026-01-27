from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .baseline import BaselinePredictor
from .run_utils import get_run_dirs
from .transformer import TransformerPredictor


def _read_texts(args: argparse.Namespace) -> List[str]:
    if args.text is not None:
        return [args.text]
    if args.texts_file is not None:
        path = Path(args.texts_file)
        if not path.exists():
            raise FileNotFoundError(f"texts file not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]


def _detect_model_dir(run_id: str) -> Path:
    model_root, _ = get_run_dirs(run_id)
    if not model_root.exists():
        raise FileNotFoundError(f"run_id not found: {run_id}")

    baseline_dir = model_root / "baseline"
    distilbert_dir = model_root / "distilbert"

    if baseline_dir.exists():
        return baseline_dir
    if distilbert_dir.exists():
        return distilbert_dir
    raise FileNotFoundError(f"no model artifacts found for run_id: {run_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a trained NLP model.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--text")
    parser.add_argument("--texts-file")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = _read_texts(args)
    if not texts:
        raise ValueError("no input texts provided")

    model_dir = _detect_model_dir(args.run_id)
    if model_dir.name == "baseline":
        predictor = BaselinePredictor.load(model_dir)
        results = predictor.predict(texts)
    else:
        predictor = TransformerPredictor.load(model_dir)
        results = predictor.predict(texts, batch_size=args.batch_size)

    for result in results:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
