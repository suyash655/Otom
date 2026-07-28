"""
Centralized configuration for ML module.

This file contains all constants and configuration parameters
used across training, inference, and preprocessing.
"""
import os
from pathlib import Path

# ── Project Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_DIR = DATA_DIR / "raw" / "Otoscopic_Data"
SPLITS_FILE = DATA_DIR / "splits.json"
FEATURES_DIR = DATA_DIR / "features"
TDA_FEATURES_FILE = FEATURES_DIR / "tda_features_clean.npy"
TDA_SCALER_FILE = FEATURES_DIR / "tda_scaler.npy"

# ── Model Architecture Constants ───────────────────────────────────────────────
IMAGE_SIZE = 224
TDA_FEATURE_DIM = 13
NUM_CLASSES = 6

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Class Configuration (must match backend and frontend) ─────────────────────
CLASS_NAMES = [
    "Acute Otitis Media",    # idx 0
    "Cerumen Impaction",     # idx 1
    "Chronic Otitis Media",  # idx 2
    "Myringosclerosis",      # idx 3
    "Normal",                # idx 4
    "Other",                 # idx 5
]

# Folder to class mapping (for raw data organization)
FOLDER_TO_CLASS = {
    # Primary classes — 1:1 mapping
    "Acute Otitis Media":   "Acute Otitis Media",
    "Cerumen Impaction":    "Cerumen Impaction",
    "Chronic Otitis Media": "Chronic Otitis Media",
    "Myringosclerosis":     "Myringosclerosis",
    "Normal":               "Normal",
    # Collapsed into Other
    "Otitis Externa":        "Other",
    "Tympanoskleros":        "Other",
    "Ear Ventilation Tube":  "Other",
    "Pseudo Membranes":      "Other",
    "Foreign Object Ear":    "Other",
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ── Class Weights for Weighted Loss ─────────────────────────────────────────────
# Formula: w_c = max(counts) / count_c, then normalised so mean(w) = 1.0
# MEDICAL-SAFETY: AOM and Chronic OM weights are intentionally high
CLASS_WEIGHTS = [
    0.49,    # idx 0  Acute Otitis Media
    0.48,    # idx 1  Cerumen Impaction
    0.53,    # idx 2  Chronic Otitis Media
    0.59,    # idx 3  Myringosclerosis
    0.31,    # idx 4  Normal
    3.58,    # idx 5  Other
]

# ── Training Configuration ───────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WARMUP_EPOCHS = 3
BACKBONE_LR_RATIO = 0.01  # Backbone LR is 1/100 of head LR
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
DROPOUT_RATE = 0.4

# ── Safety Thresholds ───────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.5
UNCERTAINTY_THRESHOLD = 0.15

# ── Device Configuration ───────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cuda" if "cuda" in os.getenv("DEVICE", "").lower() else "cpu")

# ── Augmentation Parameters ─────────────────────────────────────────────────────
AUGMENTATION_PARAMS = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.3,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
    },
    "translate": (0.1, 0.1),
}

# ── Checkpoint Configuration ───────────────────────────────────────────────────
CHECKPOINT_PREFIX = "hybrid_best"
KEEP_N_CHECKPOINTS = 3
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "hybrid_best.pth"

# ── XAI Configuration ─────────────────────────────────────────────────────────
XAI_METHODS = ["gradcam", "gradcam_pp", "integrated_gradients", "guided_backprop"]
MC_DROPOUT_SAMPLES = 30
INTEGRATED_GRADIENTS_STEPS = 50

# ── TDA Configuration ─────────────────────────────────────────────────────────
TDA_GROUPS = {
    "H0": "Connected components",
    "H1": "Loops & cavities",
}

TDA_COLORS = {
    "H0": "#3B82F6",  # blue
    "H1": "#EC4899",  # pink
}

# ── Data Split Configuration ───────────────────────────────────────────────────
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ── Path Utilities ─────────────────────────────────────────────────────────────
def get_default_paths():
    """Return default paths for training/inference."""
    return {
        "data_dir": str(RAW_DATA_DIR),
        "splits_file": str(SPLITS_FILE),
        "tda_features_file": str(TDA_FEATURES_FILE),
        "checkpoint": str(DEFAULT_CHECKPOINT),
    }


def ensure_directories():
    """Create necessary directories if they don't exist."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()