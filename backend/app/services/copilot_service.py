"""
Backend clinical reasoning service - imports canonical implementation from ML module.

This module re-exports clinical reasoning functions from the ML module to maintain
a single source of truth while providing backend-friendly imports.
"""

from __future__ import annotations

# Import canonical clinical reasoning implementation from ML module
from ml.xai.clinical_reasoning import (
    TDA_FEATURE_NAMES,
    TDA_FEATURE_CLINICAL,
    CLASS_REASONING,
    generate_clinical_reasoning,
)

# Re-export for backward compatibility
__all__ = [
    "TDA_FEATURE_NAMES",
    "TDA_FEATURE_CLINICAL",
    "CLASS_REASONING",
    "generate_clinical_reasoning",
]
