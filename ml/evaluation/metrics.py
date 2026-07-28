"""
XAI Explanation Quality Metrics.

Implements:
  - IoU / Dice with clinician annotation masks
  - AUC-Deletion score  (faithfulness: does removing attended regions drop confidence?)
  - AUC-Insertion score (faithfulness: does inserting attended regions restore confidence?)
  - Explanation stability under Gaussian noise
  - Feature rank consistency (Spearman correlation of TDA importance)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from scipy.stats import spearmanr


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation Overlap — IoU / Dice
# ──────────────────────────────────────────────────────────────────────────────

def binarize_heatmap(heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a normalised heatmap to a binary mask."""
    return (heatmap >= threshold).astype(np.float32)


def iou_score(
    predicted_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        predicted_mask: (H, W) binary float array.
        gt_mask:        (H, W) binary float array.

    Returns:
        IoU in [0, 1].
    """
    intersection = (predicted_mask * gt_mask).sum()
    union = np.clip(predicted_mask + gt_mask, 0, 1).sum()
    if union < 1e-8:
        return 1.0 if intersection < 1e-8 else 0.0
    return float(intersection / union)


def dice_score(
    predicted_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Compute Dice coefficient between two binary masks.

    Args:
        predicted_mask: (H, W) binary float array.
        gt_mask:        (H, W) binary float array.

    Returns:
        Dice in [0, 1].
    """
    intersection = (predicted_mask * gt_mask).sum()
    denom = predicted_mask.sum() + gt_mask.sum()
    if denom < 1e-8:
        return 1.0 if intersection < 1e-8 else 0.0
    return float(2.0 * intersection / denom)


def explanation_overlap_metrics(
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute IoU and Dice for a heatmap vs. a clinician mask.

    Args:
        heatmap:   (H, W) normalised float in [0, 1].
        gt_mask:   (H, W) binary float (1 = annotated region).
        threshold: Binarisation threshold for the heatmap.

    Returns:
        {"iou": float, "dice": float}
    """
    pred_mask = binarize_heatmap(heatmap, threshold)
    return {
        "iou":  iou_score(pred_mask, gt_mask),
        "dice": dice_score(pred_mask, gt_mask),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Deletion / Insertion (Faithfulness)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_deletion_mask(
    image: torch.Tensor,
    heatmap: np.ndarray,
    fraction: float,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    """Delete the top `fraction` of pixels by attribution score."""
    B, C, H, W = image.shape
    flat = heatmap.reshape(B, -1)  # (B, H*W)
    threshold_idx = int((1.0 - fraction) * flat.shape[1])
    threshold_val = np.sort(flat, axis=1)[:, threshold_idx : threshold_idx + 1]
    mask = (flat < threshold_val).reshape(B, 1, H, W).astype(np.float32)
    mask_t = torch.tensor(mask, dtype=image.dtype, device=image.device)
    return image * mask_t + baseline_value * (1.0 - mask_t)


def _apply_insertion_mask(
    image: torch.Tensor,
    heatmap: np.ndarray,
    fraction: float,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    """Insert the top `fraction` of pixels from the original, rest from baseline."""
    B, C, H, W = image.shape
    flat = heatmap.reshape(B, -1)
    threshold_idx = int((1.0 - fraction) * flat.shape[1])
    threshold_val = np.sort(flat, axis=1)[:, threshold_idx : threshold_idx + 1]
    mask = (flat >= threshold_val).reshape(B, 1, H, W).astype(np.float32)
    mask_t = torch.tensor(mask, dtype=image.dtype, device=image.device)
    return image * mask_t + baseline_value * (1.0 - mask_t)


def auc_deletion(
    model: nn.Module,
    images: torch.Tensor,
    tda_features: torch.Tensor,
    heatmap: np.ndarray,
    class_idx: int,
    n_steps: int = 10,
    baseline_value: float = 0.0,
) -> float:
    """Compute AUC-Deletion score.

    Progressively deletes the most important pixels and measures
    how quickly the model confidence drops. Lower AUC = better explanation
    (removing salient pixels hurts more).

    Args:
        model:          The HybridModel.
        images:         (B, 3, H, W) input images.
        tda_features:   (B, 14) TDA features.
        heatmap:        (B, H, W) attribution map (batch).
        class_idx:      Class index to track.
        n_steps:        Number of deletion levels.
        baseline_value: Pixel fill value for deleted regions.

    Returns:
        AUC-Deletion score (lower is better for a good explanation).
    """
    model.eval()
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    probs_curve = []

    with torch.no_grad():
        for frac in fractions:
            if frac < 1e-6:
                masked = images
            else:
                masked = _apply_deletion_mask(images, heatmap, frac, baseline_value)
            logits = model(masked, tda_features)
            p = torch.softmax(logits, dim=1)[:, class_idx].mean().item()
            probs_curve.append(p)

    return float(np.trapz(probs_curve, fractions))


def auc_insertion(
    model: nn.Module,
    images: torch.Tensor,
    tda_features: torch.Tensor,
    heatmap: np.ndarray,
    class_idx: int,
    n_steps: int = 10,
    baseline_value: float = 0.0,
) -> float:
    """Compute AUC-Insertion score.

    Progressively inserts the most important pixels from a baseline and
    measures how quickly confidence rises. Higher AUC = better explanation.

    Returns:
        AUC-Insertion score (higher is better for a good explanation).
    """
    model.eval()
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    probs_curve = []

    with torch.no_grad():
        for frac in fractions:
            if frac < 1e-6:
                masked = torch.full_like(images, baseline_value)
            else:
                masked = _apply_insertion_mask(images, heatmap, frac, baseline_value)
            logits = model(masked, tda_features)
            p = torch.softmax(logits, dim=1)[:, class_idx].mean().item()
            probs_curve.append(p)

    return float(np.trapz(probs_curve, fractions))


# ──────────────────────────────────────────────────────────────────────────────
# Explanation Stability
# ──────────────────────────────────────────────────────────────────────────────

def explanation_stability(
    explain_fn: Callable,
    images: torch.Tensor,
    tda_features: torch.Tensor,
    n_trials: int = 5,
    noise_sigma: float = 0.05,
    class_idx: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Measure stability of explanations under Gaussian noise.

    Adds small perturbations to the input and measures how much the
    explanation map changes (using Pearson correlation).

    Args:
        explain_fn:   Callable that returns (B, H, W) heatmap array.
        images:       (B, 3, H, W) original images.
        tda_features: (B, 14) TDA features.
        n_trials:     Number of noisy trials.
        noise_sigma:  Standard deviation of Gaussian noise added to input.
        class_idx:    Fixed class index for consistent comparison.

    Returns:
        mean_stability: Mean correlation coefficient across trials (higher = more stable).
        all_correlations: (n_trials,) array of individual correlations.
    """
    # Baseline explanation
    base_heatmap = explain_fn(images, tda_features, class_idx)
    base_flat = base_heatmap.reshape(-1)

    correlations = []
    for _ in range(n_trials):
        noise = torch.randn_like(images) * noise_sigma
        noisy_images = images + noise
        noisy_heatmap = explain_fn(noisy_images, tda_features, class_idx)
        noisy_flat = noisy_heatmap.reshape(-1)

        # Pearson correlation
        if base_flat.std() < 1e-8 or noisy_flat.std() < 1e-8:
            correlations.append(1.0)
        else:
            corr = np.corrcoef(base_flat, noisy_flat)[0, 1]
            correlations.append(float(corr))

    correlations_arr = np.array(correlations)
    return float(correlations_arr.mean()), correlations_arr


# ──────────────────────────────────────────────────────────────────────────────
# Feature Rank Consistency
# ──────────────────────────────────────────────────────────────────────────────

def feature_rank_consistency(
    importance_lists: list[np.ndarray],
) -> float:
    """Compute mean pairwise Spearman correlation of feature importance rankings.

    Args:
        importance_lists: List of (D,) importance arrays from different inputs or trials.

    Returns:
        Mean pairwise Spearman rho. Higher values indicate consistent ranking.
    """
    n = len(importance_lists)
    if n < 2:
        return 1.0

    rhos = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(importance_lists[i], importance_lists[j])
            rhos.append(float(rho))

    return float(np.mean(rhos))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: compute all metrics at once
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    model: nn.Module,
    images: torch.Tensor,
    tda_features: torch.Tensor,
    heatmap: np.ndarray,
    class_idx: int,
    explain_fn: Optional[Callable] = None,
    gt_mask: Optional[np.ndarray] = None,
    n_deletion_steps: int = 10,
    n_stability_trials: int = 5,
    noise_sigma: float = 0.05,
) -> dict:
    """Run all explanation quality metrics and return a summary dict.

    Args:
        model:                The HybridModel.
        images:               (B, 3, H, W)
        tda_features:         (B, 14)
        heatmap:              (B, H, W) Grad-CAM attribution maps.
        class_idx:            Target class index.
        explain_fn:           Callable for stability test (optional).
        gt_mask:              (H, W) clinician mask for IoU/Dice (optional).
        n_deletion_steps:     Number of steps for deletion/insertion curves.
        n_stability_trials:   Number of noisy trials for stability.
        noise_sigma:          Noise level for stability test.

    Returns:
        Dict of metric names → values.
    """
    results = {}

    # 1. Deletion / Insertion AUC
    results["auc_deletion"]  = auc_deletion(
        model, images, tda_features, heatmap, class_idx, n_deletion_steps
    )
    results["auc_insertion"] = auc_insertion(
        model, images, tda_features, heatmap, class_idx, n_deletion_steps
    )

    # 2. IoU / Dice (if clinician mask provided)
    if gt_mask is not None:
        # Use first image in batch
        single_heatmap = heatmap[0] if heatmap.ndim == 3 else heatmap
        overlap = explanation_overlap_metrics(single_heatmap, gt_mask, threshold=0.5)
        results["iou"]  = overlap["iou"]
        results["dice"] = overlap["dice"]
    else:
        results["iou"]  = None
        results["dice"] = None

    # 3. Stability
    if explain_fn is not None:
        stability, _ = explanation_stability(
            explain_fn, images, tda_features,
            n_trials=n_stability_trials, noise_sigma=noise_sigma,
            class_idx=class_idx,
        )
        results["stability"] = stability
    else:
        results["stability"] = None

    return results
