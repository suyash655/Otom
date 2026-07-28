"""
PyTorch Dataset for Otoscopic images with noise injection support.
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


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# CLASS STRUCTURE — edit this block when adding/removing/merging classes.
# Decision rationale (image counts from data/raw/Otoscopic_Data, June 2026):
#
#   KEEP (>=63 images, clinically distinct):
#     Normal              535  — anchor class
#     Cerumen Impaction   140  — distinct texture, well-represented
#     Acute Otitis Media  119  — primary clinical target
#     Chronic Otitis Media 63  — sufficient, high clinical importance
#
#   COLLAPSE -> "Other" (below 50-sample floor OR not reliably separable):
#     Otitis Externa       41  — too few; canal pathology, not TM-focused
#     Tympanoskleros       28  — same condition as Myringosclerosis (DE name)
#     Ear Ventilation Tube 16  — iatrogenic finding, not trainable
#     Pseudo Membranes     11  — not trainable
#     Foreign Object Ear    3  — not trainable
#   -> "Other" pool = 99 images total (usable catch-all)
#
# The "Other" class prevents the model from forcing an unknown pathology into
# one of the four named disease classes with false confidence.
# ---------------------------------------------------------------------------

# HARDCODED: output label order. Index positions are consumed by model.num_classes
# and all downstream code. Change only with a full retraining.
CLASS_NAMES = [
    "Acute Otitis Media",    # idx 0  — 719 images
    "Cerumen Impaction",     # idx 1  — 740 images
    "Chronic Otitis Media",  # idx 2  — 663 images
    "Myringosclerosis",      # idx 3  — 600 images
    "Normal",                # idx 4  — 1135 images
    "Other",                 # idx 5  — 99 images (collapsed classes)
]

# HARDCODED: maps every raw folder name in data/raw/Otoscopic_Data/ to one of
# the CLASS_NAMES above. Add a new folder->class entry here when the dataset
# gains new source folders. Never auto-detect from filesystem.
FOLDER_TO_CLASS = {
    # Primary classes — 1:1 mapping
    "Acute Otitis Media":   "Acute Otitis Media",
    "Cerumen Impaction":    "Cerumen Impaction",
    "Chronic Otitis Media": "Chronic Otitis Media",
    "Myringosclerosis":     "Myringosclerosis",
    "Normal":               "Normal",
    # Collapsed into Other — REASON noted per entry
    "Otitis Externa":        "Other",
    "Tympanoskleros":        "Other",
    "Ear Ventilation Tube":  "Other",
    "Pseudo Membranes":      "Other",
    "Foreign Object Ear":    "Other",
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# CLASS WEIGHTS for weighted CrossEntropyLoss.
# Formula: w_c = max(counts) / count_c, then normalised so mean(w) = 1.0
# This gives stronger correction than inverse-frequency for severe imbalance.
#
# Raw counts:  AOM=119, Cerumen=140, ChronicOM=63, Normal=535, Other=99
# max_count = 535 (Normal)
# raw:         535/119=4.496, 535/140=3.821, 535/63=8.492, 535/535=1.0, 535/99=5.404
# mean_raw = 4.643
# normalised:  each raw / mean_raw
#
# MEDICAL-SAFETY: recalculate using scripts/compute_weights.py after any dataset change.
# MEDICAL-SAFETY: AOM and Chronic OM weights are intentionally high — false negatives
#                 for these classes carry direct clinical harm (missed infection in children).
# ---------------------------------------------------------------------------
CLASS_WEIGHTS = [
    0.49,    # idx 0  Acute Otitis Media
    0.48,    # idx 1  Cerumen Impaction
    0.53,    # idx 2  Chronic Otitis Media
    0.59,    # idx 3  Myringosclerosis
    0.31,    # idx 4  Normal
    3.58,    # idx 5  Other
]
# ---------------------------------------------------------------------------


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def add_gaussian_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to a normalized tensor image."""
    if sigma <= 0:
        return image
    noise = torch.randn_like(image) * sigma
    return image + noise


def add_gaussian_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    """Apply Gaussian blur to a PIL image."""
    if kernel_size <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))


def add_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression artifacts to a PIL image."""
    if quality >= 100:
        return image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_brightness_shift(
    image: Image.Image, shift_percent: float
) -> Image.Image:
    """Shift brightness by a percentage (-1.0 to 1.0)."""
    if abs(shift_percent) < 1e-6:
        return image
    arr = np.array(image, dtype=np.float32)
    arr = arr + shift_percent * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class OtoscopyDataset(Dataset):
    """PyTorch Dataset for Otoscopic images."""

    def __init__(
        self,
        data_dir: str = None,
        splits_file: str = None,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tda_features_file: Optional[str] = None,
        noise_type: Optional[str] = None,
        noise_level: float = 0.0,
        subset_fraction: float = 1.0,
        seed: int = 42,
    ):
        # Set default paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        if data_dir is None:
            data_dir = str(project_root / "data" / "raw" / "Otoscopic_Data")
        if splits_file is None:
            splits_file = str(project_root / "data" / "splits.json")
        if tda_features_file is None:
            tda_features_file = str(project_root / "data" / "features" / "tda_features_clean.npy")
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.noise_type = noise_type
        self.noise_level = noise_level

        # Set default transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_eval_transforms()

        # Load splits
        with open(splits_file, "r") as f:
            data = json.load(f)

        self.samples = data["splits"][split]

        # Subsample for low-data experiments
        if subset_fraction < 1.0:
            np.random.seed(seed)
            n = max(1, int(len(self.samples) * subset_fraction))
            # Stratified subsampling
            by_class = {}
            for s in self.samples:
                by_class.setdefault(s["class"], []).append(s)
            subsampled = []
            for cls, items in by_class.items():
                k = max(1, int(len(items) * subset_fraction))
                indices = np.random.choice(len(items), k, replace=False)
                subsampled.extend([items[i] for i in indices])
            self.samples = subsampled

        # Load TDA features if provided
        self.tda_features = None
        self.tda_index = {}
        if tda_features_file and Path(tda_features_file).exists():
            tda_data = np.load(tda_features_file, allow_pickle=True).item()
            self.tda_features = tda_data["features"]  # (N, 14)
            self.tda_filenames = tda_data["filenames"]  # list of filenames
            # Build index: filename -> row index
            for i, fn in enumerate(self.tda_filenames):
                self.tda_index[fn] = i

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_name = sample["image"]
        # raw_class is the original folder name stored in splits.json
        raw_class = sample["class"]
        # HARDCODED mapping: translate folder name -> canonical class label
        # If an unknown folder appears, raise clearly rather than silently mislabelling.
        if raw_class not in FOLDER_TO_CLASS:
            raise KeyError(
                f"Folder '{raw_class}' is not in FOLDER_TO_CLASS. "
                f"Add it to dataset.py before training."
            )
        class_name = FOLDER_TO_CLASS[raw_class]
        label = CLASS_TO_IDX[class_name]

        # File lives under the original folder name, not the mapped class name
        img_path = self.data_dir / raw_class / img_name
        image = Image.open(img_path).convert("RGB")

        # Apply pre-tensor noise (blur, JPEG, brightness)
        if self.noise_type == "blur" and self.noise_level > 0:
            image = add_gaussian_blur(image, int(self.noise_level))
        elif self.noise_type == "jpeg" and self.noise_level < 100:
            image = add_jpeg_compression(image, int(self.noise_level))
        elif self.noise_type == "brightness":
            image = add_brightness_shift(image, self.noise_level)

        # Apply transforms (resize, normalize, etc.)
        image_tensor = self.transform(image)

        # Apply post-tensor noise (Gaussian noise on normalized tensor)
        if self.noise_type == "gaussian" and self.noise_level > 0:
            image_tensor = add_gaussian_noise(image_tensor, self.noise_level)

        result = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": img_name,
            "class_name": class_name,   # canonical mapped name (e.g. "Other")
            "raw_class": raw_class,     # original folder name, for debugging
        }

        # Add TDA features if available
        if self.tda_features is not None:
            key = f"{raw_class}/{img_name}"
            if key in self.tda_index:
                tda_idx = self.tda_index[key]
                result["tda_features"] = torch.tensor(
                    self.tda_features[tda_idx], dtype=torch.float32
                )
            else:
                # Return zeros if TDA features not found
                n_features = self.tda_features.shape[1]
                result["tda_features"] = torch.zeros(
                    n_features, dtype=torch.float32
                )

        return result


def get_dataloaders(
    data_dir: str = None,
    splits_file: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    tda_features_file: Optional[str] = None,
    subset_fraction: float = 1.0,
    noise_type: Optional[str] = None,
    noise_level: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    _pin_memory = torch.cuda.is_available()

    train_ds = OtoscopyDataset(
        data_dir=data_dir,
        splits_file=splits_file,
        split="train",
        tda_features_file=tda_features_file,
        subset_fraction=subset_fraction,
        seed=seed,
    )

    val_ds = OtoscopyDataset(
        data_dir=data_dir,
        splits_file=splits_file,
        split="val",
        tda_features_file=tda_features_file,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    test_ds = OtoscopyDataset(
        data_dir=data_dir,
        splits_file=splits_file,
        split="test",
        tda_features_file=tda_features_file,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory,
    )

    return train_loader, val_loader, test_loader