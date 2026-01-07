from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException

from src.cv.inference import load_cv_model, predict_image
from src.cv.data import CIFAR10_CLASSES


app = FastAPI(title="Applied Deep Learning - CIFAR10 API")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "models" / "cv" / "resnet18_cifar10.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
load_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    global model, load_error
    try:
        model = load_cv_model(CHECKPOINT_PATH, device=device)
        load_error = None
    except Exception as e:
        model = None
        load_error = str(e)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "model_loaded": model is not None,
        "load_error": load_error,
    }


@app.post("/predict_image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        suffix = ".jpg"

    tmp_path = None
    try:
        data = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        pred_name, probs = predict_image(model, tmp_path, device=device)

        probs_dict = {
            CIFAR10_CLASSES[i]: float(probs[i])
            for i in range(len(CIFAR10_CLASSES))
        }

        return {
            "filename": file.filename,
            "prediction": pred_name,
            "probabilities": probs_dict,
            "device": str(device),
            "checkpoint": str(CHECKPOINT_PATH),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
