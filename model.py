import torch
import torch.nn as nn

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:  # pragma: no cover - fallback for environments without torchvision
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None


class BaselineModel(nn.Module):
    """
    Lightweight binary classifier with an EfficientNet-style interface.
    Uses torchvision when available and falls back to a simple CNN otherwise.
    """

    def __init__(self, pretrained=False, dropout=0.30, num_classes=1):
        super().__init__()

        if efficientnet_b0 is not None and EfficientNet_B0_Weights is not None:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        if hasattr(self, "classifier"):
            features = self.backbone(x).flatten(1)
            return self.classifier(features)
        return self.backbone(x)