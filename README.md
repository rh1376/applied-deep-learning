# Applied Deep Learning - Computer Vision

End-to-end image classification on CIFAR-10 using PyTorch and transfer learning with a pretrained ResNet18. The project includes clean data loading, training with validation and early stopping, evaluation on the official test set, and simple inference utilities.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Feature extraction (freeze backbone):

```bash
python -m src.cv.train --epochs 10 --batch_size 64 --lr 3e-4 --seed 42 --mode feature
```

Fine-tuning (unfreeze all layers):

```bash
python -m src.cv.train --epochs 10 --batch_size 64 --lr 1e-4 --seed 42 --mode finetune
```

Artifacts:
- Model checkpoint: `models/cv/resnet18_cifar10.pt`
- Metrics: `reports/cv/metrics.json`
- Confusion matrix: `reports/cv/confusion_matrix.png`

## Inference

```python
from pathlib import Path
import torch

from src.cv.inference import load_cv_model, predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_cv_model(Path("models/cv/resnet18_cifar10.pt"), device)
pred_name, probs = predict_image(model, Path("path/to/image.jpg"), device)
print(pred_name)
```
