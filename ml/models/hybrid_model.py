"""
Hybrid CNN+TDA Model — ResNet-18 features concatenated with TDA features.

Experiment C: The core novel architecture.
CNN backbone (512-d) + TDA features (14-d) → MLP → classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class HybridModel(nn.Module):
    """
    Hybrid CNN+TDA model for otoscopy classification.

    Architecture:
      - CNN backbone: ResNet-18 pretrained → 512-d features
      - TDA input: 14-d topological features
      - Concatenation: 526-d
      - MLP head: Linear(526, 256) → ReLU → Dropout → Linear(256, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 6,
        tda_feature_dim: int = 13,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.tda_feature_dim = tda_feature_dim

        # CNN backbone: ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.cnn_feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # TDA feature normalization (LayerNorm — works for any batch size,
        # unlike BatchNorm1d which crashes with batch_size=1)
        self.tda_norm = nn.LayerNorm(tda_feature_dim)

        # Combined feature dimension
        combined_dim = self.cnn_feature_dim + tda_feature_dim

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def extract_cnn_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CNN features from images."""
        return self.backbone(images)

    def forward(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (B, 3, 224, 224) image tensor
            tda_features: (B, 14) TDA feature tensor
        Returns:
            logits: (B, num_classes)
        """
        # Extract CNN features
        cnn_feats = self.extract_cnn_features(images)  # (B, 512)

        # Normalize TDA features
        tda_feats = self.tda_norm(tda_features)  # (B, 14)

        # Concatenate
        combined = torch.cat([cnn_feats, tda_feats], dim=1)  # (B, 526)

        # Classify
        logits = self.classifier(combined)  # (B, num_classes)

        return logits

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        n_forward: int = 10,
    ):
        """MC Dropout: estimate epistemic uncertainty via multiple stochastic passes.

        Args:
            images: (B, 3, 224, 224) image tensor
            tda_features: (B, 14) TDA feature tensor
            n_forward: number of forward passes with dropout enabled

        Returns:
            mean_probs: (B, num_classes) averaged softmax probabilities
            uncertainty: (B,) per-sample uncertainty (mean std across classes)
            all_probs: (n_forward, B, num_classes) all individual predictions
        """
        # Enable dropout layers while keeping batchnorm in eval mode
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        for _ in range(n_forward):
            logits = self.forward(images, tda_features)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)

        # Restore eval mode
        self.eval()

        stacked = torch.stack(predictions)  # (n_forward, B, C)
        mean_probs = stacked.mean(dim=0)     # (B, C)
        uncertainty = stacked.std(dim=0).mean(dim=1)  # (B,)

        return mean_probs, uncertainty, stacked

    def get_all_features(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
    ):
        """Return CNN features, TDA features, combined features, and logits."""
        cnn_feats = self.extract_cnn_features(images)
        tda_feats = self.tda_norm(tda_features)
        combined = torch.cat([cnn_feats, tda_feats], dim=1)
        logits = self.classifier(combined)
        return cnn_feats, tda_feats, combined, logits

    def get_gradcam_target_layer(self) -> torch.nn.Module:
        """Return the target convolutional layer for Grad-CAM.
        
        Uses the last BasicBlock in layer4 of ResNet-18, which produces
        the most semantically rich feature maps before global average pooling.
        """
        return self.backbone.layer4[-1]

    def get_attention_maps(self, images: torch.Tensor) -> dict:
        """Extract intermediate feature map activations for attention analysis.
        
        Returns activation maps from each ResNet stage for multi-scale
        attention visualization.
        """
        activations = {}
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        activations["layer1"] = x.detach()
        x = self.backbone.layer2(x)
        activations["layer2"] = x.detach()
        x = self.backbone.layer3(x)
        activations["layer3"] = x.detach()
        x = self.backbone.layer4(x)
        activations["layer4"] = x.detach()
        
        return activations


class HybridModelForONNX(nn.Module):
    """Wrapper for ONNX export with separate image and TDA inputs."""

    def __init__(self, hybrid_model: HybridModel):
        super().__init__()
        self.model = hybrid_model

    def forward(self, image: torch.Tensor, tda_features: torch.Tensor):
        return self.model(image, tda_features)
