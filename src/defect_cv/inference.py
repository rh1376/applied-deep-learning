from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn

from src.defect_cv.data import get_transforms
from src.defect_cv.model import DefectCNN_Base, DefectCNN_Small

DEFECT_CLASSES = [
"Crazing",
"Inclusion",
"Patches",
"Pitted_Surface",
"Rolled-in_Scale",
"Scratches",
]

def load_defect_model(
    checkpoint_path: Path, device: torch.device, variant: str = "base"
    ) -> nn.Module:
    if variant not in {"small", "base"}:
        raise ValueError("variant must be one of {'small', 'base'}")

    model: nn.Module
    if variant == "small":
        model = DefectCNN_Small(num_classes=len(DEFECT_CLASSES), in_channels=3)
    else:
        model = DefectCNN_Base(num_classes=len(DEFECT_CLASSES), in_channels=3)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_defect_image(
    model: nn.Module, image_path: Path, device: torch.device
    ) -> tuple[str, list[float]]:
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(outputs.argmax(dim=1).item())
    pred_name = DEFECT_CLASSES[pred_idx]
    return pred_name, probs