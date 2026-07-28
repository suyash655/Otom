"""
Backend data ingestion service - imports canonical definitions from ML module.

This module re-exports dataset classes and configurations from the ML module
to maintain a single source of truth while providing backend-friendly imports.
"""

import json
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import canonical definitions from ML module
from ml.preprocessing.dataset import (
    OtoscopyDataset,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    FOLDER_TO_CLASS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_train_transforms,
    get_eval_transforms,
    get_dataloaders,
)

# Re-export for backward compatibility
__all__ = [
    "OtoscopyDataset",
    "CLASS_NAMES",
    "CLASS_WEIGHTS",
    "FOLDER_TO_CLASS",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_train_transforms",
    "get_eval_transforms",
    "get_dataloaders",
]