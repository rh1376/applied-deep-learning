from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

Mode = Literal["feature", "finetune"]


def create_resnet18(num_classes: int, mode: Mode = "feature") -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if mode == "feature":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return model


def get_trainable_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]
