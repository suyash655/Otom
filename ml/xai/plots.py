"""
Visualization and plotting module for XAI outputs.

Generates:
  - Grad-CAM overlay on original image
  - Integrated Gradients / Guided Backprop maps
  - TDA feature importance bar chart
  - ROC curve and confusion matrix
  - Explanation stability scatter plot
  - Deletion / Insertion curves
"""

from __future__ import annotations

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Sequence

# Custom clinical heatmap colormap (blue → yellow → red)
_CLINICAL_CMAP = LinearSegmentedColormap.from_list(
    "clinical",
    ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"],
)

CLASS_NAMES = [
    "Acute Otitis Media",
    "Cerumen Impaction",
    "Chronic Otitis Media",
    "Myringosclerosis",
    "Normal",
]

TDA_FEATURE_SHORT_NAMES = [
    "PEntropy H0", "PEntropy H1",
    "AmpBottle H0", "AmpBottle H1",
    "AmpWass H0", "AmpWass H1",
    "Betti H0 t1", "Betti H0 t2", "Betti H0 t3",
    "Betti H1 t1", "Betti H1 t2", "Betti H1 t3",
    "Landscape H0", "Landscape H1",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 4) RGBA numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )


def _denormalize_image(tensor_img: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation for display.

    Args:
        tensor_img: (3, H, W) or (H, W, 3) float32 array.

    Returns:
        (H, W, 3) uint8 array in [0, 255].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    if tensor_img.ndim == 3 and tensor_img.shape[0] == 3:
        img = tensor_img.transpose(1, 2, 0)
    else:
        img = tensor_img.copy()

    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM Overlay
# ──────────────────────────────────────────────────────────────────────────────

def plot_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    alpha: float = 0.45,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image:       (3, H, W) or (H, W, 3) float32 tensor-like.
        heatmap:     (H, W) normalised float [0, 1].
        title:       Plot title.
        alpha:       Heatmap overlay transparency.
        save_path:   Save to file if provided.
        return_base64: Return base64-encoded PNG string.

    Returns:
        Base64 string if return_base64=True, else None.
    """
    img_rgb = _denormalize_image(image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Heatmap only
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=axes[1], fraction=0.046, pad=0.04, label="Intensity"
    )

    # Overlay
    axes[2].imshow(img_rgb)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Multi-method comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_explanation_comparison(
    image: np.ndarray,
    gradcam: np.ndarray,
    gradcam_pp: np.ndarray,
    int_grads: np.ndarray,
    guided_bp: np.ndarray,
    predicted_class: str,
    confidence: float,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Side-by-side comparison of all four explanation methods.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    img_rgb = _denormalize_image(image)

    methods = [
        ("Grad-CAM",          gradcam),
        ("Grad-CAM++",        gradcam_pp),
        ("Integrated Grads",  int_grads),
        ("Guided Backprop",   guided_bp),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(
        f"Prediction: {predicted_class}  ({confidence:.1%} confidence)",
        fontsize=13, fontweight="bold"
    )

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for ax, (name, hmap) in zip(axes[1:], methods):
        ax.imshow(img_rgb)
        ax.imshow(hmap, cmap="jet", alpha=0.45)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# TDA Feature Importance Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_tda_importance(
    importance: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "TDA Feature Importance",
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Horizontal bar chart of TDA feature importance scores.

    Args:
        importance:    (14,) importance array, normalised [0, 1].
        feature_names: Optional list of feature names.
        title:         Chart title.
        save_path:     Save to file if provided.
        return_base64: Return base64 string.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    names = feature_names or TDA_FEATURE_SHORT_NAMES
    D = len(importance)
    names = names[:D]

    sorted_idx = np.argsort(importance)
    sorted_imp = importance[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    # Colour by H0 vs H1
    colors = ["#2196F3" if "H0" in n else "#E91E63" for n in sorted_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_names, sorted_imp, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance Score (0–1)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Value labels
    for bar, val in zip(bars, sorted_imp):
        if val > 0.02:
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8
            )

    legend_patches = [
        mpatches.Patch(color="#2196F3", label="H0 (connected components)"),
        mpatches.Patch(color="#E91E63", label="H1 (loops / cavities)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Deletion / Insertion Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_deletion_insertion(
    fractions: np.ndarray,
    deletion_probs: np.ndarray,
    insertion_probs: np.ndarray,
    auc_del: float,
    auc_ins: float,
    class_name: str = "",
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Plot Deletion and Insertion fidelity curves.

    Args:
        fractions:       (N,) fraction of pixels deleted/inserted.
        deletion_probs:  (N,) model confidence under deletion.
        insertion_probs: (N,) model confidence under insertion.
        auc_del:         AUC-Deletion scalar.
        auc_ins:         AUC-Insertion scalar.
        class_name:      Class name for the title.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fractions, deletion_probs,  "r-o",  markersize=4, label=f"Deletion  (AUC={auc_del:.3f})")
    ax.plot(fractions, insertion_probs, "b-s",  markersize=4, label=f"Insertion (AUC={auc_ins:.3f})")
    ax.set_xlabel("Fraction of pixels removed/inserted", fontsize=11)
    ax.set_ylabel("Model confidence", fontsize=11)
    ax.set_title(
        f"Explanation Faithfulness{' — ' + class_name if class_name else ''}",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Plot a normalised confusion matrix.

    Args:
        cm:           (C, C) integer confusion matrix.
        class_names:  List of class label strings.
        title:        Chart title.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    names = class_names or CLASS_NAMES
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=[n.split()[0] for n in names],
        yticklabels=[n.split()[0] for n in names],
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# ROC Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        all_probs:   (N, C) softmax probabilities.
        all_labels:  (N,) integer true labels.
        class_names: Class name strings.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    names = class_names or CLASS_NAMES
    n_classes = len(names)
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, color) in enumerate(zip(names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name.split()[0]} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Stability Scatter Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_stability(
    correlations: np.ndarray,
    noise_sigma: float,
    method_name: str = "Grad-CAM",
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Bar plot showing explanation stability correlations across noise trials.

    Args:
        correlations: (T,) Pearson correlation per trial.
        noise_sigma:  Noise level used.
        method_name:  XAI method name for title.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    T = len(correlations)
    trials = np.arange(1, T + 1)
    mean_corr = correlations.mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(trials, correlations, color="#4CAF50", edgecolor="white")
    ax.axhline(mean_corr, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_corr:.3f}")
    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Pearson Correlation with Baseline", fontsize=11)
    ax.set_title(
        f"{method_name} Stability (σ={noise_sigma})",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Confidence distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_confidence_bars(
    class_names: Sequence[str],
    probabilities: np.ndarray,
    predicted_idx: int,
    save_path: Optional[str] = None,
    return_base64: bool = False,
) -> Optional[str]:
    """Horizontal bar chart of class probabilities.

    Args:
        class_names:  (C,) class name strings.
        probabilities:(C,) softmax probabilities.
        predicted_idx: Index of the predicted class.

    Returns:
        Base64 PNG string if return_base64=True.
    """
    n = len(class_names)
    colors = ["#E53935" if i == predicted_idx else "#90A4AE" for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(
        [n.split(" ")[0] + ("\n" + " ".join(n.split(" ")[1:])) if len(n) > 15 else n
         for n in class_names],
        probabilities, color=colors, edgecolor="white", height=0.6
    )
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title("Prediction Confidence", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, probabilities):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    result = _fig_to_base64(fig) if return_base64 else None
    plt.close(fig)
    return result
