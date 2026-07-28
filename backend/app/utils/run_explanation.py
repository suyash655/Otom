"""
XAI Evaluation Script — run all explanations on a single image or a batch.

Usage:
  python -m src.xai.run_explanation --image path/to/image.png
  python -m src.xai.run_explanation --image path/to/image.png --checkpoint checkpoints/hybrid_best.pth
  python -m src.xai.run_explanation --batch data/raw/Otoscopic_Data --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Local imports using importlib to avoid package-level import issues ──────
import importlib.util as _ilu


def _load_module(name: str, rel_path: str):
    """Load a local module by relative path from this file's parent tree."""
    here = Path(__file__).resolve().parent
    root = here.parent.parent  # d:\otoc\
    abs_path = root / rel_path
    spec = _ilu.spec_from_file_location(name, abs_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hybrid_mod = _load_module("hybrid_model",   "src/models/hybrid_model.py")
_dataset_mod = _load_module("dataset",        "src/data/dataset.py")
_tda_mod     = _load_module("tda_extract",    "src/features/tda_extract.py")

HybridModel    = _hybrid_mod.HybridModel
CLASS_NAMES    = _dataset_mod.CLASS_NAMES
get_eval_transforms = _dataset_mod.get_eval_transforms
extract_single_array_features = _tda_mod.extract_single_array_features
get_feature_names = _tda_mod.get_feature_names

# XAI modules
from src.xai.explainer        import HybridExplainer
from src.xai.clinical_reasoning import generate_clinical_reasoning
from src.xai.metrics          import compute_all_metrics
from src.xai.plots import (
    plot_gradcam_overlay,
    plot_explanation_comparison,
    plot_tda_importance,
    plot_confidence_bars,
)


# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: torch.device) -> HybridModel:
    """Load HybridModel from checkpoint."""
    model = HybridModel(
        num_classes=len(CLASS_NAMES),
        tda_feature_dim=14,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_image(image_path: str, device: torch.device):
    """Load, transform, and prepare an image + TDA features for inference."""
    transform = get_eval_transforms(image_size=224)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Image tensor
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # TDA features
    tda_feats = extract_single_array_features(img_np, size=32)
    tda_tensor = torch.tensor(tda_feats, dtype=torch.float32).unsqueeze(0).to(device)

    return img_tensor, tda_tensor, img_np


def explain_single_image(
    image_path: str,
    checkpoint: str,
    output_dir: str,
    device: str = "cpu",
):
    """Run full XAI pipeline on a single image and save results."""
    dev = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n→ Loading model from {checkpoint}")
    model = load_model(checkpoint, dev)

    print(f"→ Preparing image: {image_path}")
    img_tensor, tda_tensor, img_np = prepare_image(image_path, dev)

    # Run all explanations
    print("→ Running XAI methods...")
    explainer = HybridExplainer(model)
    results   = explainer.explain(img_tensor, tda_tensor)

    pred_idx  = results["predicted_class"]
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    confidence = float(results["class_probs"][0, pred_idx])

    print(f"  Prediction: {pred_name}  ({confidence:.1%})")

    # Image array for plots — take first item in batch
    img_arr = img_tensor[0].cpu().numpy()

    # Plot: Grad-CAM overlay
    gradcam_path = str(output_path / "gradcam.png")
    plot_gradcam_overlay(
        img_arr,
        results["gradcam"][0],
        title=f"Grad-CAM — {pred_name}",
        save_path=gradcam_path,
    )
    print(f"  Saved Grad-CAM → {gradcam_path}")

    # Plot: All methods comparison
    comparison_path = str(output_path / "comparison.png")
    plot_explanation_comparison(
        img_arr,
        results["gradcam"][0],
        results["gradcam_pp"][0],
        results["int_grads"][0],
        results["guided_bp"][0],
        pred_name,
        confidence,
        save_path=comparison_path,
    )
    print(f"  Saved comparison → {comparison_path}")

    # Plot: TDA feature importance
    tda_path = str(output_path / "tda_importance.png")
    plot_tda_importance(
        results["tda_importance"],
        feature_names=get_feature_names(),
        title=f"TDA Feature Importance — {pred_name}",
        save_path=tda_path,
    )
    print(f"  Saved TDA importance → {tda_path}")

    # Plot: Confidence bars
    conf_path = str(output_path / "confidence.png")
    plot_confidence_bars(
        CLASS_NAMES,
        results["class_probs"][0],
        pred_idx,
        save_path=conf_path,
    )
    print(f"  Saved confidence bars → {conf_path}")

    # XAI metrics (deletion / insertion)
    print("→ Computing faithfulness metrics...")
    metrics = compute_all_metrics(
        model,
        img_tensor,
        tda_tensor,
        results["gradcam"],
        pred_idx,
        explain_fn=lambda imgs, tdas, ci: explainer.gradcam.generate(imgs, tdas, ci),
        n_deletion_steps=10,
        n_stability_trials=3,
        noise_sigma=0.05,
    )
    print(f"  AUC-Deletion:  {metrics['auc_deletion']:.4f}")
    print(f"  AUC-Insertion: {metrics['auc_insertion']:.4f}")
    if metrics["stability"] is not None:
        print(f"  Stability:     {metrics['stability']:.4f}")

    # Clinical reasoning
    tda_vals = tda_tensor[0].cpu().numpy()
    reasoning = generate_clinical_reasoning(
        predicted_class=pred_name,
        confidence=confidence,
        heatmap=results["gradcam"][0],
        tda_values=tda_vals,
        tda_importance=results["tda_importance"],
    )

    # Save JSON report
    report = {
        "image_path":      image_path,
        "predicted_class": pred_name,
        "confidence":      confidence,
        "class_probs":     {CLASS_NAMES[i]: float(p) for i, p in enumerate(results["class_probs"][0])},
        "tda_importance":  {n: float(v) for n, v in zip(get_feature_names(), results["tda_importance"])},
        "metrics":         {k: (float(v) if v is not None else None) for k, v in metrics.items()},
        "clinical_reasoning": reasoning,
    }
    report_path = str(output_path / "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report → {report_path}")

    # Print clinical summary
    print("\n── Clinical Reasoning ──────────────────────────────────────────")
    print(reasoning["summary"])
    print(reasoning["attention"])
    for finding in reasoning["tda_findings"]:
        print(f"  • {finding[:120]}...")
    print(f"  {reasoning['confidence_note']}")
    print("─────────────────────────────────────────────────────────────────\n")

    return report


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run XAI explanation on otoscopic images.")
    parser.add_argument("--image",      type=str, help="Path to a single image file.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hybrid_best.pth")
    parser.add_argument("--output-dir", type=str, default="assets/examples")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    if not args.image:
        parser.print_help()
        sys.exit(1)

    explain_single_image(
        image_path=args.image,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
