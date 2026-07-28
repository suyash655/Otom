"""
CNN Baseline Model — ResNet-18 pretrained on ImageNet.

Experiment A: Fine-tuned ResNet-18 for otoscopy classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBaseline(nn.Module):
    """ResNet-18 baseline for otoscopy classification."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Feature dimension from ResNet-18
        self.feature_dim = self.backbone.fc.in_features  # 512

        # Replace classification head
        self.backbone.fc = nn.Identity()

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-d features from ResNet-18 backbone."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → logits."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def get_features_and_logits(self, x: torch.Tensor):
        """Return both features and logits (for SHAP/UMAP analysis)."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return features, logits
