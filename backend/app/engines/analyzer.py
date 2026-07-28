"""
Hybrid Model Engine - Imports and re-exports the HybridModel from ML module.

This module serves as a bridge between the backend and ML layers,
allowing the backend to import the model architecture from the
canonical ML module location.
"""
from __future__ import annotations

# Import the canonical HybridModel from the ML module
from ml.models.hybrid_model import HybridModel, HybridModelForONNX

__all__ = ["HybridModel", "HybridModelForONNX"]
