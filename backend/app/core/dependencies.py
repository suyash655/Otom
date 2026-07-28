"""
FastAPI dependencies for the OtoScope AI backend.

Provides:
  - Model/explainer singleton loader (get_model, get_explainer)
  - Image transform singleton
"""
from __future__ import annotations

from typing import Optional

import torch

from backend.app.core.config import (
    ROOT,
    DEVICE,
    BEST_CHECKPOINT,
    DUMMY_CHECKPOINT,
    logger,
)

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_MODEL = None
_EXPLAINER = None
_CKPT_CLASS_NAMES: list = []
_TRANSFORM = None





def get_transform():
    """Return the eval image transform singleton."""
    global _TRANSFORM
    if _TRANSFORM is None:
        from backend.app.services.data_ingestion import get_eval_transforms
        _TRANSFORM = get_eval_transforms(image_size=224)
    return _TRANSFORM


def get_class_names() -> list:
    """Return class names from the loaded checkpoint (or config fallback)."""
    get_model()  # ensure model is loaded
    return _CKPT_CLASS_NAMES


def get_model():
    """Return the model singleton, loading from checkpoint on first call."""
    global _MODEL, _EXPLAINER, _CKPT_CLASS_NAMES

    if _MODEL is not None:
        return _MODEL

    # Resolve checkpoint
    ckpt_path = BEST_CHECKPOINT if BEST_CHECKPOINT.exists() else DUMMY_CHECKPOINT
    if not ckpt_path.exists():
        raise RuntimeError(
            "No model checkpoint found. "
            "Run: python scripts/setup/train.py --dummy  to create one."
        )

    logger.info(f"Loading model from {ckpt_path}")

    # Load modules
    from backend.app.engines.analyzer import HybridModel
    from backend.app.services.data_ingestion import CLASS_NAMES

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    tda_feature_dim = int(state["tda_norm.weight"].shape[0])
    num_classes = int(state["classifier.3.weight"].shape[0])

    if tda_feature_dim != 14:
        logger.warning(f"Checkpoint tda_feature_dim={tda_feature_dim} (config=14)")
    if num_classes != len(CLASS_NAMES):
        logger.warning(f"Checkpoint num_classes={num_classes} (config={len(CLASS_NAMES)})")

    model = HybridModel(
        num_classes=num_classes,
        tda_feature_dim=tda_feature_dim,
        pretrained=False,
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    _CKPT_CLASS_NAMES = ckpt.get("class_names") or CLASS_NAMES

    from backend.app.services.ml_service import HybridExplainer
    _MODEL = model
    _EXPLAINER = HybridExplainer(model)

    logger.info(
        f"Model loaded — num_classes={num_classes}, "
        f"tda_feature_dim={tda_feature_dim}, "
        f"class_names={_CKPT_CLASS_NAMES}"
    )
    return _MODEL


def get_explainer():
    """Return the explainer singleton."""
    get_model()  # ensures _EXPLAINER is initialised
    return _EXPLAINER
