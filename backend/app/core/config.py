"""
Centralized configuration management for the backend.

Uses environment variables with sensible defaults.
"""
import logging
import os
from pathlib import Path
from typing import Optional

# ── Project Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ROOT = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# ── API Server ─────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_URL = f"http://{HOST}:{PORT}"

# ── Model Configuration ─────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    str(CHECKPOINTS_DIR / "hybrid_best.pth")
)
DEVICE = os.getenv("DEVICE", "cuda" if "cuda" in os.getenv("DEVICE", "").lower() else "cpu")

# ── Safety Thresholds ───────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.5"))
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.15"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", str(LOGS_DIR / "audit.jsonl"))
logger = logging.getLogger("backend")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# ── ML Module Paths ───────────────────────────────────────────────────────────
ML_MODULE_PATH = PROJECT_ROOT / "ml"
MODELS_PATH = ML_MODULE_PATH / "models"
PREPROCESSING_PATH = ML_MODULE_PATH / "preprocessing"
TRAINING_PATH = ML_MODULE_PATH / "training"
INFERENCE_PATH = ML_MODULE_PATH / "inference"
TDA_PATH = ML_MODULE_PATH / "tda"
XAI_PATH = ML_MODULE_PATH / "xai"

# ── Data Paths ───────────────────────────────────────────────────────────────
RAW_DATA_DIR = DATA_DIR / "raw" / "Otoscopic_Data"
SPLITS_FILE = DATA_DIR / "splits.json"
FEATURES_DIR = DATA_DIR / "features"
TDA_FEATURES_FILE = FEATURES_DIR / "tda_features_clean.npy"
BEST_CHECKPOINT = CHECKPOINTS_DIR / "hybrid_best.pth"
DUMMY_CHECKPOINT = CHECKPOINTS_DIR / "hybrid_dummy.pth"

# ── Model Architecture Constants ───────────────────────────────────────────────
IMAGE_SIZE = 224
TDA_FEATURE_DIM = 13
NUM_CLASSES = 6  # Normal, AOM, Cerumen, Chronic OM, Myringosclerosis, Other

# ── Class Names (must match dataset.py) ────────────────────────────────────────
CLASS_NAMES = [
    "Acute Otitis Media",
    "Cerumen Impaction", 
    "Chronic Otitis Media",
    "Myringosclerosis",
    "Normal",
    "Other",
]

# ── XAI Configuration ─────────────────────────────────────────────────────────
XAI_METHODS = ["gradcam", "gradcam_pp", "integrated_gradients", "guided_backprop"]
MC_DROPOUT_SAMPLES = int(os.getenv("MC_DROPOUT_SAMPLES", "30"))

# ── CORS Configuration ───────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ── Ensure directories exist ───────────────────────────────────────────────────
def ensure_directories():
    """Create necessary directories if they don't exist."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()