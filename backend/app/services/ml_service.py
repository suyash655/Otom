"""
Backend ML service - imports canonical XAI implementations from ML module.

This module re-exports XAI explainer classes from the ML module to maintain
a single source of truth while providing backend-friendly imports.
"""

from __future__ import annotations

# Import canonical XAI implementations from ML module
from ml.xai.explainer import (
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    GuidedBackpropagation,
    TDAFeatureImportance,
    HybridExplainer,
)

# Re-export for backward compatibility
__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "IntegratedGradients",
    "GuidedBackpropagation",
    "TDAFeatureImportance",
    "HybridExplainer",
]
