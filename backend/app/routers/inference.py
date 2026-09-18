"""
Inference router: /health, /predict, /explain
"""
from __future__ import annotations

import importlib.util as _ilu
import time

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.config import DEVICE, ROOT, logger
from backend.app.core.dependencies import get_class_names, get_explainer, get_model, get_transform
from backend.app.schemas.prediction import ExplainResponse, PredictionResponse
from backend.app.services import xai_service

router = APIRouter()



# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health():
    """Health check."""
    from backend.app.services.data_ingestion import CLASS_NAMES
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


# ── /predict ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded otoscopic image.

    Upload a PNG/JPEG image and receive:
    - predicted class name + index
    - prediction confidence
    - full class probability distribution
    - MC Dropout uncertainty estimate
    """
    try:
        model = get_model()
        transform = get_transform()
        image_bytes = await file.read()
        img_tensor, tda_tensor, _ = xai_service.prepare_inputs(image_bytes, model, transform)
    except Exception as exc:
        logger.exception("Error loading image")
        raise HTTPException(status_code=422, detail=f"Image loading error: {exc}")

    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img_tensor, tda_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        mean_probs, uncertainty, _ = model.predict_with_uncertainty(
            img_tensor, tda_tensor, n_forward=10
        )
        processing_time_ms = (time.perf_counter() - t0) * 1000
        ckpt_class_names = get_class_names()

        pred_idx = int(probs.argmax())
        pred_name = ckpt_class_names[pred_idx] if pred_idx < len(ckpt_class_names) else str(pred_idx)
        confidence = float(probs[pred_idx])
        unc_value = float(uncertainty[0].item())

        low_confidence = confidence < 0.5
        uncertain = unc_value > 0.15
        if low_confidence or uncertain:
            logger.warning(
                f"Low-confidence prediction: class={pred_name} "
                f"conf={confidence:.3f} unc={unc_value:.3f}"
            )

        xai_service.audit_log(
            image_bytes, pred_name, confidence, unc_value, "/predict",
            processing_time_ms=processing_time_ms,
        )

        return PredictionResponse(
            predicted_class=pred_name,
            predicted_index=pred_idx,
            confidence=confidence,
            class_probs={ckpt_class_names[i]: float(p) for i, p in enumerate(probs)},
            uncertainty=unc_value,
            low_confidence=low_confidence,
            uncertain=uncertain,
            processing_time_ms=round(processing_time_ms, 2),
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# ── /explain ──────────────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(file: UploadFile = File(...)):
    """Classify an image and return full XAI explanations.

    Returns:
    - Prediction with confidence
    - Grad-CAM, Grad-CAM++, Integrated Gradients, Guided Backpropagation
      heatmaps as base64-encoded PNG images
    - TDA feature importance scores
    - Clinical reasoning narrative
    """
    try:
        model = get_model()
        transform = get_transform()
        image_bytes = await file.read()
        img_tensor, tda_tensor, img_np = xai_service.prepare_inputs(image_bytes, model, transform)
    except Exception as exc:
        logger.exception("Error loading image")
        raise HTTPException(status_code=422, detail=f"Image loading error: {exc}")

    try:
        t0 = time.perf_counter()
        xai = get_explainer()
        results = xai.explain(img_tensor, tda_tensor)

        ckpt_class_names = get_class_names()
        pred_idx = results["predicted_class"]
        pred_name = ckpt_class_names[pred_idx] if pred_idx < len(ckpt_class_names) else str(pred_idx)
        probs = results["class_probs"][0]
        confidence = float(probs[pred_idx])
        tda_imp = results["tda_importance"]
        tda_vals = tda_tensor[0].detach().cpu().numpy()
        img_arr = img_tensor[0].detach().cpu().numpy()

        mean_p, unc, _ = model.predict_with_uncertainty(img_tensor, tda_tensor, n_forward=10)
        unc_value = float(unc[0].item())

        low_confidence = confidence < 0.5
        uncertain = unc_value > 0.15
        if low_confidence or uncertain:
            logger.warning(
                f"Low-confidence explanation: class={pred_name} "
                f"conf={confidence:.3f} unc={unc_value:.3f}"
            )

        # Plots
        b64_gradcam = xai_service.render_heatmap_b64(img_arr, results["gradcam"][0], "Grad-CAM")
        b64_gradcam_pp = xai_service.render_heatmap_b64(img_arr, results["gradcam_pp"][0], "Grad-CAM++")
        b64_int_grads = xai_service.render_heatmap_b64(img_arr, results["int_grads"][0], "Integrated Gradients")
        b64_guided_bp = xai_service.render_heatmap_b64(img_arr, results["guided_bp"][0], "Guided Backpropagation")

        b64_comparison = xai_service.render_comparison_b64(
            img_arr,
            results["gradcam"][0], results["gradcam_pp"][0],
            results["int_grads"][0], results["guided_bp"][0],
            pred_name, confidence,
        )

        from ml.xai.clinical_reasoning import TDA_FEATURE_NAMES
        feature_names = TDA_FEATURE_NAMES

        b64_tda = xai_service.render_tda_b64(tda_imp, feature_names, pred_name)
        b64_conf = xai_service.render_confidence_b64(ckpt_class_names, probs, pred_idx)

        # Clinical reasoning
        from ml.xai.clinical_reasoning import generate_clinical_reasoning
        reasoning = generate_clinical_reasoning(
            predicted_class=pred_name,
            confidence=confidence,
            heatmap=results["gradcam"][0],
            tda_values=tda_vals,
            tda_importance=tda_imp,
        )
        reasoning["research_disclaimer"] = (
            "RESEARCH ONLY. This clinical reasoning text is algorithmically "
            "generated from feature lookup tables. It has not been validated "
            "by clinicians and must not be used for diagnosis or treatment."
        )

        processing_time_ms = (time.perf_counter() - t0) * 1000
        xai_service.audit_log(
            image_bytes, pred_name, confidence, unc_value, "/explain",
            processing_time_ms=processing_time_ms,
        )

        return ExplainResponse(
            predicted_class=pred_name,
            predicted_index=pred_idx,
            confidence=confidence,
            class_probs={ckpt_class_names[i]: float(p) for i, p in enumerate(probs)},
            uncertainty=unc_value,
            low_confidence=low_confidence,
            uncertain=uncertain,
            processing_time_ms=round(processing_time_ms, 2),
            heatmap_gradcam=b64_gradcam,
            heatmap_gradcam_pp=b64_gradcam_pp,
            heatmap_int_grads=b64_int_grads,
            heatmap_guided_bp=b64_guided_bp,
            chart_comparison=b64_comparison,
            chart_tda=b64_tda,
            chart_confidence=b64_conf,
            tda_importance={n: float(v) for n, v in zip(feature_names, tda_imp)},
            tda_values={n: float(v) for n, v in zip(feature_names, tda_vals)},
            clinical_reasoning=reasoning,
        )
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=f"Explanation error: {exc}")
