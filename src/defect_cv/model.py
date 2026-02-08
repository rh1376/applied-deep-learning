from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation if activation is not None else F.relu
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class DefectCNN_Small(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3) -> None:
        super().__init__()
        act = F.silu

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, activation=act, dropout_p=0.0),
            ConvBlock(32, 32, activation=act, dropout_p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, activation=act, dropout_p=0.05),
            ConvBlock(64, 64, activation=act, dropout_p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, activation=act, dropout_p=0.1),
            ConvBlock(128, 128, activation=act, dropout_p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, activation=act, dropout_p=0.1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, 224, 224).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.head(x)
        return x
    
class DefectCNN_Base(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3) -> None:
        super().__init__()
        act = F.silu

        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, activation=act, dropout_p=0.0),
            ConvBlock(32, 32, activation=act, dropout_p=0.0),
        )

        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, activation=act, dropout_p=0.05),
            ConvBlock(64, 64, activation=act, dropout_p=0.05),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, activation=act, dropout_p=0.1),
            ConvBlock(128, 128, activation=act, dropout_p=0.1),
            ConvBlock(128, 128, activation=act, dropout_p=0.1),
        )

        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, activation=act, dropout_p=0.15),
            ConvBlock(256, 256, activation=act, dropout_p=0.15),
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 384, activation=act, dropout_p=0.2),
            ConvBlock(384, 384, activation=act, dropout_p=0.2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(192, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, 224, 224).
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(name: str, num_classes: int = 6, in_channels: int = 3) -> nn.Module:
        """
        Factory for model variants.

        Args:
            name: "small" or "base"
            num_classes: number of output classes
            in_channels: number of input channels
        """
        registry: Dict[str, Callable[[], nn.Module]] = {
            "small": lambda: DefectCNN_Small(num_classes=num_classes, in_channels=in_channels),
            "base": lambda: DefectCNN_Base(num_classes=num_classes, in_channels=in_channels),
        }
        key = name.strip().lower()
        if key not in registry:
            raise ValueError(f"Unknown model name '{name}'. Supported: {sorted(registry.keys())}")
        return registry[key]()