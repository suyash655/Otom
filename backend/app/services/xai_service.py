"""
XAI service: audit logging, input preparation, and explanation orchestration.

Keeps all heavy glue logic out of the router layer.
"""
from __future__ import annotations

import io
import json
import hashlib
import datetime
import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from backend.app.core.config import ROOT, DEVICE, AUDIT_LOG_PATH, logger




# ── Audit logging ─────────────────────────────────────────────────────────────

def audit_log(
    image_bytes: bytes,
    pred_class: str,
    confidence: float,
    uncertainty: float,
    endpoint: str,
    app_version: str = "1.0.0",
    processing_time_ms: float = None,
) -> None:
    """Append a prediction audit record to logs/audit.jsonl."""
    try:
        image_hash = hashlib.sha256(image_bytes[:1024]).hexdigest()
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "image_sha256_1k": image_hash,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "model_version": app_version,
            "endpoint": endpoint,
        }
        if processing_time_ms is not None:
            record["processing_time_ms"] = round(processing_time_ms, 2)
            
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.warning("Failed to write audit log", exc_info=True)


import functools

# ── Input preparation ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=128)
def _get_cached_tda(image_hash: str, img_bytes: bytes, tda_dim: int) -> list:
    from ml.tda.tda_extract import extract_single_array_features
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    tda_feats = extract_single_array_features(img_np, size=32)
    if len(tda_feats) != tda_dim:
        logger.warning(
            f"TDA extractor returned {len(tda_feats)} features but model "
            f"expects {tda_dim}. Adjusting."
        )
        tda_feats = tda_feats[:tda_dim]
    return tda_feats

def prepare_inputs(image_bytes: bytes, model, transform) -> Tuple:
    """Convert raw image bytes → (img_tensor, tda_tensor, img_np).
    
    TDA feature dim is read from the loaded model so it always matches
    the checkpoint, regardless of what tda_feature_dim it was trained with.
    """
    tda_dim = model.tda_feature_dim
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    tda_feats = _get_cached_tda(image_hash, image_bytes, tda_dim)

    tda_tensor = torch.tensor(tda_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return img_tensor, tda_tensor, img_np


# ── Plot helpers ──────────────────────────────────────────────────────────────

def render_heatmap_b64(img_arr: np.ndarray, hmap: np.ndarray, method: str) -> str:
    from backend.app.utils import plot_helpers as plots
    return plots.plot_gradcam_overlay(img_arr, hmap, title=method, return_base64=True)


def render_comparison_b64(img_arr, gcam, gcam_pp, ints, guided, pred_name, confidence) -> str:
    from backend.app.utils import plot_helpers as plots
    return plots.plot_explanation_comparison(
        img_arr, gcam, gcam_pp, ints, guided, pred_name, confidence, return_base64=True
    )


def render_tda_b64(tda_imp, feature_names, pred_name) -> str:
    from backend.app.utils import plot_helpers as plots
    return plots.plot_tda_importance(
        tda_imp,
        feature_names=feature_names,
        title=f"TDA Feature Importance — {pred_name}",
        return_base64=True,
    )


def render_confidence_b64(class_names, probs, pred_idx) -> str:
    from backend.app.utils import plot_helpers as plots
    return plots.plot_confidence_bars(class_names, probs, pred_idx, return_base64=True)
