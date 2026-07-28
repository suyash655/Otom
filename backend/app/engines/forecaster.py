"""
Forecaster engine — training utilities for the HybridModel.

This module re-exports training helpers from scripts/train.py so they can be
imported as part of the backend engine layer (e.g., for future background
training tasks via Celery / RQ).

For standalone CLI usage, run:
  python scripts/train.py --train
  python scripts/train.py --eval
  python scripts/train.py --dummy
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export from the training script
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("train", ROOT / "scripts" / "train.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_criterion = _mod.build_criterion
build_scheduler = _mod.build_scheduler
per_class_metrics = _mod.per_class_metrics
train_one_epoch = _mod.train_one_epoch
evaluate = _mod.evaluate
run_dummy_check = _mod.run_dummy_check
run_training = _mod.run_training
run_eval = _mod.run_eval

__all__ = [
    "build_criterion",
    "build_scheduler",
    "per_class_metrics",
    "train_one_epoch",
    "evaluate",
    "run_dummy_check",
    "run_training",
    "run_eval",
]
