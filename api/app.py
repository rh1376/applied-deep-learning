from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Tuple, Any

from src.cv.inference import load_cv_model, predict_image
from src.cv.data import CIFAR10_CLASSES
from src.defect_cv.inference import (
    DEFECT_CLASSES,
    load_defect_model,
    predict_defect_image,
)

from src.nlp.run_utils import get_run_dirs
from src.nlp.baseline import BaselinePredictor
from src.nlp.transformer import TransformerPredictor
from src.rag_helpdesk.schemas import RAGQueryRequest, RAGQueryResponse, Citation
from src.rag_helpdesk.ingest import load_rag_engine

app = FastAPI(title="Applied Deep Learning")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "models" / "cv" / "resnet18_cifar10.pt"
DEFECT_CHECKPOINT_PATH = REPO_ROOT / "models" / "defect_cv" / "best.pt"
RAG_INDEX_DIR = REPO_ROOT / "data" / "helpdesk_kb" / "index"
RAG_INDEX_FILE = RAG_INDEX_DIR / "index.faiss"
RAG_METADATA_FILE = RAG_INDEX_DIR / "metadata.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
load_error: str | None = None
defect_model = None
defect_load_error: str | None = None
defect_default_variant: str = os.getenv("DEFECT_MODEL", "base")

# ---------- NLP globals ----------
nlp_default_run_id: str | None = os.getenv("NLP_RUN_ID")  # optional
nlp_default_model: str = os.getenv("NLP_MODEL", "auto")   # auto | baseline | distilbert
nlp_load_error: str | None = None
_nlp_cache: Dict[Tuple[str, str], Any] = {}  # (run_id, model_type) -> predictor

# ---------- RAG globals ----------
rag_embed_model: str = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
rag_llm_model: str = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
try:
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
except ValueError:
    rag_top_k = 5
rag_device: str = "cuda" if torch.cuda.is_available() else "cpu"
rag_engine = None
rag_load_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:

    global defect_model, defect_load_error
    global model, load_error, nlp_load_error
    global rag_engine, rag_load_error
    try:
        model = load_cv_model(CHECKPOINT_PATH, device=device)
        load_error = None
    except Exception as e:
        model = None
        load_error = str(e)

    try:
        defect_model = load_defect_model(
            DEFECT_CHECKPOINT_PATH, device=device, variant=defect_default_variant
        )
        defect_load_error = None
    except Exception as e:
        defect_model = None
        defect_load_error = str(e)


    # Optional: load a default NLP predictor if NLP_RUN_ID is set
    if nlp_default_run_id:
        try:
            _ = _get_nlp_predictor(run_id=nlp_default_run_id, model_type=nlp_default_model)
            nlp_load_error = None
        except Exception as e:
            nlp_load_error = str(e)

    try:
        rag_engine = load_rag_engine(
            index_dir=RAG_INDEX_DIR,
            embedding_model_name=rag_embed_model,
            llm_model_name=rag_llm_model,
            top_k=rag_top_k,
            device=rag_device,
        )
        rag_load_error = None
    except Exception as e:
        rag_engine = None
        rag_load_error = str(e)


@app.get("/health")
def health():
    rag_index_exists = RAG_INDEX_FILE.exists() and RAG_METADATA_FILE.exists()

    return {
        "status": "ok",
        "device": str(device),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "model_loaded": model is not None,
        "load_error": load_error,
        "defect_checkpoint_exists": DEFECT_CHECKPOINT_PATH.exists(),
        "defect_model_loaded": defect_model is not None,
        "defect_load_error": defect_load_error,
        "defect_default_variant": defect_default_variant,
        "nlp_default_run_id": nlp_default_run_id,
        "nlp_default_model": nlp_default_model,
        "nlp_load_error": nlp_load_error,
        "rag_index_exists": rag_index_exists,
        "rag_loaded": rag_engine is not None,
        "rag_load_error": rag_load_error,
        "rag_embed_model": rag_embed_model,
        "rag_llm_model": rag_llm_model,
        "rag_top_k": rag_top_k,
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


@app.post("/predict_defect_image")
async def predict_defect_image_endpoint(file: UploadFile = File(...)):
    if defect_model is None:
        raise HTTPException(
            status_code=500, detail=f"Defect model not loaded: {defect_load_error}"
        )

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

        pred_name, probs = predict_defect_image(defect_model, tmp_path, device=device)

        probs_dict = {
            DEFECT_CLASSES[i]: float(probs[i]) for i in range(len(DEFECT_CLASSES))
        }

        return {
            "filename": file.filename,
            "prediction": pred_name,
            "probabilities": probs_dict,
            "device": str(device),
            "checkpoint": str(DEFECT_CHECKPOINT_PATH),
            "variant": defect_default_variant,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass




# =========================
# NLP API (added)
# =========================

def _detect_nlp_model_dir(run_id: str) -> Path:
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


def _get_nlp_predictor(run_id: str, model_type: str = "auto"):
    """
    model_type: auto | baseline | distilbert
    Caches predictors in-memory to avoid reloading per request.
    """
    model_type = (model_type or "auto").lower()
    if model_type not in {"auto", "baseline", "distilbert"}:
        raise ValueError("model must be one of: auto, baseline, distilbert")

    if model_type == "auto":
        model_dir = _detect_nlp_model_dir(run_id)
        model_type = model_dir.name  # baseline|distilbert
    else:
        model_root, _ = get_run_dirs(run_id)
        model_dir = model_root / model_type
        if not model_dir.exists():
            raise FileNotFoundError(f"model artifacts not found: {model_dir}")

    key = (run_id, model_type)
    if key in _nlp_cache:
        return _nlp_cache[key]

    if model_type == "baseline":
        predictor = BaselinePredictor.load(model_dir)
    else:
        predictor = TransformerPredictor.load(model_dir)

    _nlp_cache[key] = predictor
    return predictor


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    run_id: Optional[str] = None
    model: Literal["auto", "baseline", "distilbert"] = "auto"
    batch_size: int = Field(32, ge=1, le=256)


@app.post("/predict_text")
def predict_text_endpoint(req: NLPRequest):
    """
    Predict on a single text using either baseline or distilbert artifacts saved under:
      models/nlp/<run_id>/{baseline|distilbert}

    If req.run_id is not provided, uses env NLP_RUN_ID (if set).
    """
    run_id = req.run_id or nlp_default_run_id
    if not run_id:
        raise HTTPException(
            status_code=400,
            detail="run_id is required (either in request body or via NLP_RUN_ID env var).",
        )

    try:
        predictor = _get_nlp_predictor(run_id=run_id, model_type=req.model)

        # baseline/distilbert predictors both support predict_one(text)
        if hasattr(predictor, "predict_one"):
            result = predictor.predict_one(req.text)
        else:
            # fallback
            result = predictor.predict([req.text], batch_size=req.batch_size)[0]

        return {
            "run_id": run_id,
            "model": req.model,
            "result": result,  # {"label":..., "score":..., "probs":{...}}
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP prediction failed: {e}")


@app.post("/rag_helpdesk_query", response_model=RAGQueryResponse)
def rag_helpdesk_query_endpoint(req: RAGQueryRequest):
    if rag_engine is None:
        detail = rag_load_error or "RAG engine failed to load"
        raise HTTPException(status_code=500, detail=f"RAG engine not loaded: {detail}")

    effective_top_k = req.top_k if req.top_k is not None else rag_top_k
    if effective_top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    try:
        original_top_k = rag_engine.retriever.top_k
        try:
            rag_engine.retriever.top_k = effective_top_k
            result = rag_engine.answer(req.query)
        finally:
            rag_engine.retriever.top_k = original_top_k

        citations = [
            Citation(
                score=float(item.get("score", 0.0)),
                source_path=str(item.get("source_path", "")),
                chunk_id=str(item.get("chunk_id", "")),
                doc_id=str(item.get("doc_id", "")),
            )
            for item in result.get("citations", [])
        ]

        response = RAGQueryResponse(
            answer=str(result.get("answer", "")),
            citations=citations,
            context_used=[str(x) for x in result.get("context_used", [])],
            model=rag_llm_model,
            top_k=effective_top_k,
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")
