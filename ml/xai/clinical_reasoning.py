"""
Clinical Reasoning module — maps XAI outputs to human-readable clinical text.

Provides:
  - Attention-region descriptions from Grad-CAM heatmaps
  - TDA feature interpretation (topological meaning)
  - Combined clinical narrative
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# Feature names from tda_extract.py
TDA_FEATURE_NAMES = [
    "PersEntropy_H0",
    "PersEntropy_H1",
    "AmpBottleneck_H0",
    "AmpBottleneck_H1",
    "AmpWasserstein_H0",
    "AmpWasserstein_H1",
    "Betti_H0_t1",
    "Betti_H0_t2",
    "Betti_H0_t3",
    "Betti_H1_t1",
    "Betti_H1_t2",
    "Betti_H1_t3",
    "Landscape_H0",
    "Landscape_H1",
]

# Clinical descriptions for each TDA feature
TDA_FEATURE_CLINICAL = {
    "PersEntropy_H0": (
        "H0 persistence entropy measures the complexity of connected components "
        "in the image. High values indicate irregular tissue distribution, "
        "which can reflect inflamed or disrupted tympanic membrane structure."
    ),
    "PersEntropy_H1": (
        "H1 persistence entropy captures the topological complexity of loops/cycles. "
        "Elevated H1 entropy suggests abnormal cavity structures or fluid pockets "
        "behind the eardrum, consistent with middle ear effusion."
    ),
    "AmpBottleneck_H0": (
        "H0 bottleneck amplitude (maximum component lifetime) reflects the "
        "dominant connected region in the image. High values suggest a large "
        "consolidated opacity, consistent with cerumen or exudate."
    ),
    "AmpBottleneck_H1": (
        "H1 bottleneck amplitude captures the most persistent loop structure. "
        "High values indicate a dominant circular feature, potentially the "
        "tympanic membrane annulus or a perforation boundary."
    ),
    "AmpWasserstein_H0": (
        "H0 Wasserstein amplitude measures overall complexity of connected regions. "
        "Elevated values indicate multiple distinct tissue regions, common in "
        "chronic otitis media with granulation tissue."
    ),
    "AmpWasserstein_H1": (
        "H1 Wasserstein amplitude quantifies total loop persistence energy. "
        "High values suggest multiple ring-like structures, which can correspond "
        "to myringosclerosis (calcium plaques) on the tympanic membrane."
    ),
    "Betti_H0_t1": (
        "H0 Betti number at low threshold (early filtration) counts the number of "
        "initially connected components. High counts suggest fragmented or "
        "heterogeneous tissue — a marker of inflammatory changes."
    ),
    "Betti_H0_t2": (
        "H0 Betti number at mid threshold provides a stability measure of tissue "
        "connectivity. Changes from the low-threshold count indicate intermediate-"
        "scale structural variation in the membrane."
    ),
    "Betti_H0_t3": (
        "H0 Betti number at high threshold (late filtration) represents the final "
        "connected structure. A value >1 here suggests multi-region pathology "
        "such as bilateral involvement or complex scarring."
    ),
    "Betti_H1_t1": (
        "H1 Betti number at low threshold counts early-forming loops. Elevated "
        "values can indicate vessel patterns or early-stage cavity formation "
        "associated with acute otitis media."
    ),
    "Betti_H1_t2": (
        "H1 Betti number at mid threshold tracks loop persistence. This feature "
        "is particularly discriminative for chronic otitis media, where long-"
        "persisting perforations create stable topological loops."
    ),
    "Betti_H1_t3": (
        "H1 Betti number at high threshold (persistent loops only). "
        "A non-zero value here strongly suggests a tympanic membrane perforation "
        "or significant structural defect."
    ),
    "Landscape_H0": (
        "H0 persistence landscape area summarises the total 'weight' of connected "
        "components. Large values indicate widespread tissue involvement, "
        "consistent with extensive cerumen impaction."
    ),
    "Landscape_H1": (
        "H1 persistence landscape area quantifies overall loop complexity. "
        "High values correlate with irregular membrane topology, seen in "
        "myringosclerosis or healed perforation scars."
    ),
}

# Class-specific clinical context
CLASS_CLINICAL_CONTEXT = {
    "Acute Otitis Media": (
        "Acute Otitis Media (AOM) is a bacterial or viral middle ear infection. "
        "The tympanic membrane typically appears bulging, erythematous, and "
        "opaque. Key indicators include loss of light reflex and reduced mobility."
    ),
    "Cerumen Impaction": (
        "Cerumen Impaction is a buildup of earwax obstructing the ear canal. "
        "The otoscopic view shows a brownish, occlusive mass with varying texture "
        "depending on the consistency (hard vs. soft wax)."
    ),
    "Chronic Otitis Media": (
        "Chronic Otitis Media (CSOM) involves persistent middle ear infection often "
        "with perforation. Key features include a visible hole in the tympanic "
        "membrane, mucopurulent discharge, and middle ear wall visibility."
    ),
    "Normal": (
        "A normal tympanic membrane appears pearly-grey, translucent with a visible "
        "light reflex (cone of light), intact cone shape, and clearly visible "
        "malleus handle. No fluid, perforation, or discolouration is present."
    ),
    "Other": (
        "The model could not confidently classify this image into one of the four "
        "primary categories. Possible findings include Otitis Externa, Tympanosclerosis, "
        "Ear Ventilation Tube, Pseudo Membranes, or Foreign Object. "
        "Clinical examination by an ENT specialist is required."
    ),
}

# Spatial attention region descriptions
_QUADRANT_LABELS = {
    (0, 0): "superior-anterior quadrant",
    (0, 1): "superior-posterior quadrant",
    (1, 0): "inferior-anterior quadrant",
    (1, 1): "inferior-posterior quadrant",
}


def describe_attention_region(heatmap: np.ndarray) -> str:
    """Describe where the Grad-CAM attention is focusing.

    Divides the heatmap into four quadrants and identifies the dominant region(s).

    Args:
        heatmap: (H, W) float array in [0, 1].

    Returns:
        Human-readable region description.
    """
    H, W = heatmap.shape
    hm = heatmap / (heatmap.sum() + 1e-8)

    quadrants = {
        (0, 0): hm[: H // 2, : W // 2].sum(),
        (0, 1): hm[: H // 2, W // 2 :].sum(),
        (1, 0): hm[H // 2 :, : W // 2].sum(),
        (1, 1): hm[H // 2 :, W // 2 :].sum(),
    }

    # Sort by attention weight
    sorted_q = sorted(quadrants.items(), key=lambda kv: kv[1], reverse=True)
    top_q = sorted_q[0]

    # Check if attention is relatively concentrated
    concentration = top_q[1]
    region = _QUADRANT_LABELS[top_q[0]]

    if concentration > 0.45:
        focus = "strongly"
    elif concentration > 0.30:
        focus = "primarily"
    else:
        focus = "diffusely"

    # Check if centre vs periphery
    h_pad = H // 4
    w_pad = W // 4
    centre_mass = hm[h_pad : H - h_pad, w_pad : W - w_pad].sum()
    if centre_mass > 0.55:
        location = "the central tympanic membrane region"
    else:
        location = f"the {region} of the tympanic membrane"

    return f"Attention is {focus} concentrated on {location}."


def describe_tda_features(
    tda_values: np.ndarray,
    importance: np.ndarray,
    top_k: int = 3,
) -> list[str]:
    """Generate clinical descriptions for the top-k most important TDA features.

    Args:
        tda_values:  (D,) raw TDA feature values.
        importance:  (D,) importance scores (0–1).
        top_k:       Number of features to describe.

    Returns:
        List of clinical strings, one per top feature.
    """
    D = len(TDA_FEATURE_NAMES)
    top_indices = np.argsort(importance)[::-1][:top_k]
    descriptions = []

    for idx in top_indices:
        if idx >= D:
            continue
        name = TDA_FEATURE_NAMES[idx]
        value = float(tda_values[idx]) if idx < len(tda_values) else 0.0
        imp   = float(importance[idx])
        clinical = TDA_FEATURE_CLINICAL.get(name, "")

        value_level = "high" if value > 0.6 else ("moderate" if value > 0.3 else "low")

        descriptions.append(
            f"**{name}** (importance: {imp:.2f}, value level: {value_level}): "
            f"{clinical}"
        )

    return descriptions


def generate_clinical_reasoning(
    predicted_class: str,
    confidence: float,
    heatmap: np.ndarray,
    tda_values: np.ndarray,
    tda_importance: np.ndarray,
    top_k_tda: int = 3,
) -> dict:
    """Generate a full clinical reasoning report.

    Args:
        predicted_class: String class name.
        confidence:      Prediction confidence (0–1).
        heatmap:         (H, W) Grad-CAM heatmap.
        tda_values:      (14,) raw TDA feature vector.
        tda_importance:  (14,) importance scores.
        top_k_tda:       Number of TDA features to highlight.

    Returns:
        Dict with keys:
          - summary:         One-sentence clinical summary
          - attention:       Attention region description
          - tda_findings:    List of TDA clinical descriptions
          - class_context:   Background on the predicted condition
          - confidence_note: Confidence interpretation
    """
    # Confidence interpretation
    if confidence >= 0.85:
        conf_note = (
            f"The model is highly confident ({confidence:.1%}). "
            "This prediction warrants clinical review for confirmation."
        )
    elif confidence >= 0.60:
        conf_note = (
            f"Moderate confidence ({confidence:.1%}). "
            "Consider clinical correlation and possible differential diagnoses."
        )
    else:
        conf_note = (
            f"Low confidence ({confidence:.1%}). "
            "The image may be ambiguous or partially occluded. "
            "Expert clinical assessment is strongly recommended."
        )

    attention_desc  = describe_attention_region(heatmap)
    tda_descriptions = describe_tda_features(tda_values, tda_importance, top_k=top_k_tda)
    class_context   = CLASS_CLINICAL_CONTEXT.get(predicted_class, "")

    summary = (
        f"The hybrid TDA model predicts **{predicted_class}** "
        f"with {confidence:.1%} confidence based on combined "
        f"visual and topological analysis of the otoscopic image."
    )

    return {
        "summary":        summary,
        "attention":      attention_desc,
        "tda_findings":   tda_descriptions,
        "class_context":  class_context,
        "confidence_note": conf_note,
        "research_disclaimer": (
            "RESEARCH ONLY. This clinical reasoning text is algorithmically "
            "generated from feature lookup tables. It has not been validated "
            "by clinicians and must not be used for diagnosis or treatment."
        ),
    }
