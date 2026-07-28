"""
XAI Explainer module for the Hybrid CNN+TDA model.

Implements:
  - GradCAM (standard Grad-CAM on ResNet-18 layer4)
  - GradCAMPlusPlus (Grad-CAM++ improved version)
  - IntegratedGradients (pixel-level attribution)
  - GuidedBackpropagation (sharp edge-level attribution)
  - TDA feature importance via input perturbation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_map(m: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D attribution map to [0, 1]."""
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def _resize_map(m: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D map to (H, W) using bilinear interpolation."""
    import PIL.Image as _PILImage
    h, w = target_hw
    pil = _PILImage.fromarray((m * 255).astype(np.uint8))
    pil = pil.resize((w, h), _PILImage.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Standard Grad-CAM for the HybridModel CNN backbone.

    Hooks into ``model.backbone.layer4[-1]`` (the last ResNet block) and
    computes a class-discriminative localisation map.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_gradcam_target_layer()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []

    def _register_hooks(self):
        def _fwd_hook(_, __, output):
            self._activations = output.detach()

        def _bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(
            self.target_layer.register_forward_hook(_fwd_hook)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(_bwd_hook)
        )

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmaps.

        Args:
            images:       (B, 3, 224, 224) tensor
            tda_features: (B, 14) tensor
            class_idx:    Target class. If None, uses predicted class.

        Returns:
            heatmaps: (B, H, W) float32 array in [0, 1], same spatial size as images.
        """
        self.model.eval()
        self._register_hooks()
        try:
            images = images.requires_grad_(True)
            logits = self.model(images, tda_features)

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1)[0].item())

            # Scalar score for the target class (summed over batch for simplicity)
            score = logits[:, class_idx].sum()
            self.model.zero_grad()
            score.backward()

            # activations: (B, C, h, w)
            act = self._activations  # (B, C, h, w)
            grad = self._gradients   # (B, C, h, w)

            # Global-average-pool the gradients → (B, C)
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

            # Weighted combination + ReLU
            cam = (weights * act).sum(dim=1)   # (B, h, w)
            cam = F.relu(cam)                   # keep positive attributions

            B, _, H, W = images.shape
            heatmaps = []
            cam_np = cam.detach().cpu()
            for b in range(B):
                m = cam_np[b].numpy()
                m = _normalize_map(m)
                m = _resize_map(m, (H, W))
                heatmaps.append(m)

            return np.stack(heatmaps)  # (B, H, W)
        finally:
            self._remove_hooks()
            self._activations = None
            self._gradients = None


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM++
# ──────────────────────────────────────────────────────────────────────────────

class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ — improved pixel importance estimation.

    Uses second-order gradient weights for sharper localisation.
    """

    def generate(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        self._register_hooks()
        try:
            images = images.requires_grad_(True)
            logits = self.model(images, tda_features)

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1)[0].item())

            score = logits[:, class_idx].sum()
            self.model.zero_grad()
            score.backward()

            act = self._activations  # (B, C, h, w)
            grad = self._gradients   # (B, C, h, w)

            # Grad-CAM++ weights
            grad_sq  = grad ** 2
            grad_cub = grad ** 3
            alpha_denom = 2.0 * grad_sq + act * grad_cub
            alpha_denom = torch.where(
                alpha_denom != 0,
                alpha_denom,
                torch.ones_like(alpha_denom),
            )
            alpha = grad_sq / alpha_denom  # (B, C, h, w)

            # Normalised weights (sum over spatial)
            relu_grad = F.relu(grad)       # only positive gradients
            weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)

            cam = (weights * act).sum(dim=1)  # (B, h, w)
            cam = F.relu(cam)

            B, _, H, W = images.shape
            heatmaps = []
            cam_np = cam.detach().cpu()
            for b in range(B):
                m = cam_np[b].numpy()
                m = _normalize_map(m)
                m = _resize_map(m, (H, W))
                heatmaps.append(m)

            return np.stack(heatmaps)
        finally:
            self._remove_hooks()
            self._activations = None
            self._gradients = None


# ──────────────────────────────────────────────────────────────────────────────
# Integrated Gradients
# ──────────────────────────────────────────────────────────────────────────────

class IntegratedGradients:
    """Integrated Gradients for pixel-level attribution.

    Approximates the integral of gradients along a straight path from
    a baseline (black image) to the input image.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def generate(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute Integrated Gradients attribution maps.

        Args:
            images:       (B, 3, H, W)
            tda_features: (B, 14)
            class_idx:    Target class index (uses prediction if None)
            baseline:     Reference input, defaults to all-zeros.

        Returns:
            attributions: (B, H, W) absolute attribution, normalised [0, 1].
        """
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(images)

        # Determine class index from first sample prediction
        if class_idx is None:
            with torch.no_grad():
                logits = self.model(images, tda_features)
            class_idx = int(logits.argmax(dim=1)[0].item())

        # Build interpolation steps
        alphas = torch.linspace(0, 1, self.n_steps, device=images.device)
        integrated_grads = torch.zeros_like(images)

        for alpha in alphas:
            # Create a fresh leaf tensor for each step so .grad is populated
            interp = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
            logits = self.model(interp, tda_features)
            score = logits[:, class_idx].sum()
            self.model.zero_grad()
            score.backward()
            if interp.grad is not None:
                integrated_grads += interp.grad.detach()

        # Scale by (input - baseline) / n_steps
        integrated_grads = integrated_grads * (images - baseline).detach() / self.n_steps

        # Aggregate across colour channels → absolute sum
        attr = integrated_grads.detach().abs().sum(dim=1)  # (B, H, W)

        B, _, H, W = images.shape
        heatmaps = []
        for b in range(B):
            m = attr[b].numpy()
            m = _normalize_map(m)
            heatmaps.append(m)

        return np.stack(heatmaps)


# ──────────────────────────────────────────────────────────────────────────────
# Guided Backpropagation
# ──────────────────────────────────────────────────────────────────────────────

class GuidedBackpropagation:
    """Guided Backpropagation — sharp edge-level attribution.

    Modifies the ReLU backward pass to only propagate positive gradients
    through positive activations.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._relu_handles: list = []

    def _register_relu_hooks(self):
        def _guided_relu_hook(_, grad_input, grad_output):
            # Keep only positive gradients through ReLU (guided backprop rule)
            # grad_input[0] is the gradient flowing into the ReLU input
            g = grad_input[0]
            if g is not None:
                return (torch.clamp(g, min=0.0),)
            return grad_input

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # Use register_backward_hook (not full_backward_hook) to
                # avoid the inplace-view conflict with ResNet's inplace ReLUs
                self._relu_handles.append(
                    module.register_backward_hook(_guided_relu_hook)
                )

    def _remove_relu_hooks(self):
        for h in self._relu_handles:
            h.remove()
        self._relu_handles.clear()

    def generate(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Guided Backpropagation attribution maps.

        Returns:
            attributions: (B, H, W) absolute attribution, normalised [0, 1].
        """
        self.model.eval()
        self._register_relu_hooks()
        try:
            images_req = images.detach().requires_grad_(True)
            logits = self.model(images_req, tda_features)

            if class_idx is None:
                class_idx = int(logits.argmax(dim=1)[0].item())

            score = logits[:, class_idx].sum()
            self.model.zero_grad()
            score.backward()

            grads = images_req.grad  # (B, 3, H, W)
            attr = grads.detach().abs().sum(dim=1)  # (B, H, W)

            B = images.shape[0]
            heatmaps = []
            for b in range(B):
                m = attr[b].cpu().numpy()
                m = _normalize_map(m)
                heatmaps.append(m)

            return np.stack(heatmaps)
        finally:
            self._remove_relu_hooks()


# ──────────────────────────────────────────────────────────────────────────────
# TDA Feature Importance
# ──────────────────────────────────────────────────────────────────────────────

class TDAFeatureImportance:
    """Compute TDA feature importance via input perturbation (occlusion).

    Masks each TDA feature to zero one at a time and measures the drop
    in prediction probability for the target class.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-feature importance scores.

        Args:
            images:       (B, 3, H, W)
            tda_features: (B, D)
            class_idx:    Target class index.

        Returns:
            importance: (D,) array of importance scores, normalised [0, 1].
        """
        self.model.eval()
        D = tda_features.shape[1]

        with torch.no_grad():
            base_logits = self.model(images, tda_features)
            probs = torch.softmax(base_logits, dim=1)

            if class_idx is None:
                class_idx = int(probs.argmax(dim=1)[0].item())

            baseline_prob = probs[:, class_idx].mean().item()

        importances = []
        with torch.no_grad():
            for d in range(D):
                perturbed = tda_features.clone()
                perturbed[:, d] = 0.0
                perturbed_logits = self.model(images, perturbed)
                perturbed_probs = torch.softmax(perturbed_logits, dim=1)
                p = perturbed_probs[:, class_idx].mean().item()
                importances.append(max(0.0, baseline_prob - p))

        imp = np.array(importances, dtype=np.float32)
        # Normalise to [0, 1]
        if imp.max() > 1e-8:
            imp = imp / imp.max()
        return imp


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

class HybridExplainer:
    """High-level explainer combining all XAI methods for the HybridModel."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradcam      = GradCAM(model)
        self.gradcam_pp   = GradCAMPlusPlus(model)
        self.int_grads    = IntegratedGradients(model, n_steps=50)
        self.guided_bp    = GuidedBackpropagation(model)
        self.tda_imp      = TDAFeatureImportance(model)

    def explain(
        self,
        images: torch.Tensor,
        tda_features: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> dict:
        """Run all XAI methods and return a results dict.

        Returns:
            {
                "gradcam":       (B, H, W) ndarray,
                "gradcam_pp":    (B, H, W) ndarray,
                "int_grads":     (B, H, W) ndarray,
                "guided_bp":     (B, H, W) ndarray,
                "tda_importance":(D,) ndarray,
                "predicted_class": int,
                "class_probs":   (B, C) ndarray,
            }
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images, tda_features)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        if class_idx is None:
            class_idx = int(probs[0].argmax())

        results = {
            "predicted_class": class_idx,
            "class_probs":     probs,
            "gradcam":         self.gradcam.generate(images, tda_features, class_idx),
            "gradcam_pp":      self.gradcam_pp.generate(images, tda_features, class_idx),
            "int_grads":       self.int_grads.generate(images, tda_features, class_idx),
            "guided_bp":       self.guided_bp.generate(images, tda_features, class_idx),
            "tda_importance":  self.tda_imp.compute(images, tda_features, class_idx),
        }
        return results
