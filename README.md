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

. ├── api/ FastAPI application ├── src/ │ ├── cv/ Computer Vision
(training + inference) │ └── nlp/ NLP (training + inference utilities)
├── models/ │ ├── cv/ Trained CV models │ └── nlp/ Trained NLP models
(per run_id) ├── reports/ Metrics, plots, and experiment outputs ├──
requirements.txt └── README.txt

  -------
  Setup
  -------

Create and activate a virtual environment, then install dependencies:

python -m venv .venv source .venv/bin/activate (Windows: .venv) pip
install -r requirements.txt

  ----------------------
  Computer Vision (CV)
  ----------------------

Training

The CV module trains a ResNet18 model on the CIFAR-10 dataset.

Feature extraction (frozen backbone): python -m src.cv.train –mode
feature

Fine-tuning (all layers trainable): python -m src.cv.train –mode
finetune

Artifacts

After training, the following artifacts are produced:

-   Model checkpoint: models/cv/resnet18_cifar10.pt
-   Training metrics and plots: reports/cv/

Inference (Python)

from pathlib import Path import torch from src.cv.inference import
load_cv_model, predict_image

device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)
model = load_cv_model(Path(“models/cv/resnet18_cifar10.pt”), device)

label, probs = predict_image(model, Path(“example.jpg”), device)
print(label)

  -----------------------------------
  Natural Language Processing (NLP)
  -----------------------------------

The NLP module supports two approaches:

-   Baseline: A lightweight classical model (e.g. TF-IDF + Logistic
    Regression)
-   DistilBERT: A fine-tuned transformer model for text classification

Training

Baseline model: python -m src.nlp.train_baseline

DistilBERT model: python -m src.nlp.train_distilbert

Artifacts and run_id

Each NLP training run creates a directory under:

models/nlp//

Example: models/nlp/20260127-215237-distilbert/ ├── baseline/ └──
distilbert/

The folder name (run_id) uniquely identifies a training run and is
required for inference and API requests.

Inference (Python)

from src.nlp.run_utils import get_run_dirs from src.nlp.baseline import
BaselinePredictor from src.nlp.transformer import TransformerPredictor

run_id = “20260127-215237-distilbert” model_root, _ =
get_run_dirs(run_id)

bert = TransformerPredictor.load(model_root / “distilbert”) result =
bert.predict_one(“This movie was surprisingly good.”) print(result)

  ---------------
  API (FastAPI)
  ---------------

Run the API

From the repository root:

uvicorn api.app:app –reload

Swagger UI

Once the server is running, open:

http://127.0.0.1:8000/docs

You can test both CV and NLP inference endpoints directly from the
browser.

  -----------------
  NLP API Example
  -----------------

Request body:

{ “text”: “The movie had strong performances but the story lost focus
halfway through.”, “run_id”: “20260127-215237-distilbert”, “model”:
“distilbert” }

Example response:

{ “label”: “pos”, “score”: 0.63, “probs”: { “neg”: 0.37, “pos”: 0.63 } }

  -------
  Notes
  -------

-   Use –reload only for development.
-   GPU acceleration is automatically used if a compatible CUDA-enabled
    PyTorch build is installed.
-   The project is structured to be easily extended with additional
    models or tasks.

Summary

This project demonstrates an end-to-end applied deep learning workflow,
from training and evaluation to deployment-ready inference using
FastAPI.
