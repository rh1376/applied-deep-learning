from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from .data import CIFAR10_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from .model import create_resnet18


def _inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_cv_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint["num_classes"])
    model = create_resnet18(num_classes=num_classes, mode="finetune")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: nn.Module, image_path: Path, device: torch.device
) -> Tuple[str, list[float]]:
    transform = _inference_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.argmax(torch.tensor(probs)))
    pred_name = CIFAR10_CLASSES[pred_idx]
    return pred_name, probs
