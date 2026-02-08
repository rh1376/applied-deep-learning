Applied Deep Learning (CV & NLP)

This repository contains an applied deep learning project with two main
components:

-   Computer Vision (CV): Image classification on the CIFAR-10 dataset
    using PyTorch and transfer learning.
-   Natural Language Processing (NLP): Text classification using both a
    classical baseline model and a transformer-based model (DistilBERT).

The project also provides a FastAPI-based inference API that allows
testing both CV and NLP models via HTTP requests and Swagger UI.

--------------------------------
Project Structure (High-level)
--------------------------------

.
├── api/                     FastAPI application
├── src/
│   ├── cv/                  Computer Vision (training + inference)
│   ├── nlp/                 NLP (training + inference utilities)
│   └── defect_cv/           Industrial Defect Detection (custom CNN)
├── models/
│   ├── cv/                  Trained CV models
│   ├── nlp/                 Trained NLP models (per run_id)
│   └── defect_cv/           Trained industrial defect models
├── reports/                 Metrics, confusion matrices, experiment outputs
├── requirements.txt
└── README.md

-------
Setup
-------

Create and activate a virtual environment, then install dependencies:

python -m venv .venv
source .venv/bin/activate   (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

----------------------
Computer Vision (CV)
----------------------

Training

The CV module trains a ResNet18 model on the CIFAR-10 dataset.

Feature extraction (frozen backbone):
python -m src.cv.train --mode feature

Fine-tuning (all layers trainable):
python -m src.cv.train --mode finetune

Artifacts

After training, the following artifacts are produced:

-   Model checkpoint: models/cv/resnet18_cifar10.pt
-   Training metrics and plots: reports/cv/

Inference (Python)

from pathlib import Path
import torch
from src.cv.inference import load_cv_model, predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cv_model(Path("models/cv/resnet18_cifar10.pt"), device)

label, probs = predict_image(model, Path("example.jpg"), device)
print(label)

-----------------------------------
Natural Language Processing (NLP)
-----------------------------------

The NLP module supports two approaches:

-   Baseline: A lightweight classical model (e.g. TF-IDF + Logistic Regression)
-   DistilBERT: A fine-tuned transformer model for text classification

Training

Baseline model:
python -m src.nlp.train_baseline

DistilBERT model:
python -m src.nlp.train_distilbert

Artifacts and run_id

Each NLP training run creates a directory under:

models/nlp/<run_id>/

Example:
models/nlp/20260127-215237-distilbert/
├── baseline/
└── distilbert/

The folder name (run_id) uniquely identifies a training run and is
required for inference and API requests.

Inference (Python)

from src.nlp.run_utils import get_run_dirs
from src.nlp.baseline import BaselinePredictor
from src.nlp.transformer import TransformerPredictor

run_id = "20260127-215237-distilbert"
model_root, _ = get_run_dirs(run_id)

bert = TransformerPredictor.load(model_root / "distilbert")
result = bert.predict_one("This movie was surprisingly good.")
print(result)

---------------
API (FastAPI)
---------------

Run the API

From the repository root:

uvicorn api.app:app --reload

Swagger UI

Once the server is running, open:

http://127.0.0.1:8000/docs

You can test CV, NLP, and Industrial Defect Detection inference endpoints
directly from the browser.

-----------------
NLP API Example
-----------------

Request body:

{
  "text": "The movie had strong performances but the story lost focus halfway through.",
  "run_id": "20260127-215237-distilbert",
  "model": "distilbert"
}

Example response:

{
  "label": "pos",
  "score": 0.63,
  "probs": {
    "neg": 0.37,
    "pos": 0.63
  }
}

-------
Notes
-------

-   Use --reload only for development.
-   GPU acceleration is automatically used if a compatible CUDA-enabled
    PyTorch build is installed.
-   The project is structured to be easily extended with additional
    models or tasks.

Summary

This project demonstrates an end-to-end applied deep learning workflow,
from training and evaluation to deployment-ready inference using FastAPI.

=====================================================================
Industrial Defect Detection (Applied Computer Vision Project)
=====================================================================

## Problem Statement

Industrial surface defect detection is a common computer vision task in
manufacturing domains such as steel, fabric, and PCB inspection.
The goal is to automatically classify surface images into defect types
to reduce manual inspection costs and improve quality control.

This module implements a fully custom, end-to-end deep learning pipeline
for industrial defect classification.

## Dataset

- Dataset: NEU Surface Defect Classification (NEU-CLS)
- Number of classes: 6
- Defect categories:
  - Crazing
  - Inclusion
  - Patches
  - Pitted_Surface
  - Rolled-in_Scale
  - Scratches

Images are grayscale and converted to 3-channel tensors during
preprocessing to align with standard CNN pipelines.

## Model Architecture

Two custom CNN variants are implemented from scratch:

- **DefectCNN_Small**
  - Lightweight baseline model
  - Faster training and inference
  - Used for sanity checks and ablation studies

- **DefectCNN_Base**
  - Stage-based CNN architecture
  - Progressive downsampling and increasing channel depth
  - Batch Normalization and Dropout for regularization
  - Global Average Pooling before classification head

Both models are implemented without pretrained backbones to explicitly
demonstrate CNN design and training principles.

## Training Details

- Framework: PyTorch
- Loss function: Weighted CrossEntropyLoss (to handle class imbalance)
- Optimizer: AdamW
- Learning rate scheduler: CosineAnnealingLR
- Mixed Precision Training: Automatic Mixed Precision (AMP)
- Reproducibility:
  - Fixed random seeds
  - Deterministic cuDNN settings

## Results

- Validation Accuracy: ~98.8–99%
- Stable convergence observed after early epochs
- Confusion matrix analysis used to inspect per-class performance
- Best-performing checkpoint saved automatically during training

Artifacts generated:
- models/defect_cv/best.pt
- reports/defect_cv/metrics.json
- reports/defect_cv/confusion_matrix.txt

## Inference & API

Inference is supported both offline and via FastAPI.

### FastAPI Endpoint

- **POST /predict_defect_image**
- Accepts an image file upload
- Returns predicted defect class and per-class probabilities

Example response:

{
  "prediction": "Scratches",
  "probabilities": {
    "Crazing": 0.0012,
    "Inclusion": 0.0008,
    "Patches": 0.0021,
    "Pitted_Surface": 0.0034,
    "Rolled-in_Scale": 0.0115,
    "Scratches": 0.9810
  }
}

## How to Run (Quick Start)

Training:

python -m src.defect_cv.train --model base --epochs 20 --batch_size 32

Run API:

uvicorn api.app:app --reload

Swagger UI:

http://127.0.0.1:8000/docs

## Future Improvements

- Grad-CAM visualization for model explainability
- Model optimization for faster inference
- Dockerized deployment
- Extension to defect localization and segmentation
