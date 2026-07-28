"""
TDA Feature Extraction using Persistent Homology.

Extracts 14 topological features per image:
  - Persistence Entropy (H0, H1) → 2
  - Amplitude-Bottleneck (H0, H1) → 2
  - Amplitude-Wasserstein (H0, H1) → 2
  - Betti numbers at 3 filtration thresholds (H0, H1) → 6
  - Persistence Landscape area (H0, H1) → 2
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple

# Backend detection is deferred to _detect_backend() so that merely
# importing this module never raises an ImportError.
_TDA_BACKEND: Optional[str] = None


def _detect_backend() -> str:
    """Detect and cache which TDA backend is available."""
    global _TDA_BACKEND
    if _TDA_BACKEND is not None:
        return _TDA_BACKEND
    try:
        import gtda.homology  # noqa: F401
        _TDA_BACKEND = "giotto"
    except ImportError:
        try:
            import ripser  # noqa: F401
            _TDA_BACKEND = "ripser"
        except ImportError:
            raise ImportError(
                "Neither giotto-tda nor ripser+persim is installed.\n"
                "Install one with:\n"
                "  pip install ripser persim          (recommended, pure-Python)\n"
                "  pip install giotto-tda==0.6.0      (requires Python <=3.10)"
            )
    return _TDA_BACKEND





def preprocess_image_for_tda(
    image: np.ndarray, size: int = 32
) -> np.ndarray:
    """Convert image to grayscale and resize to small bitmap for TDA."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    # Resize to small grid
    from PIL import Image as PILImage
    pil_gray = PILImage.fromarray(gray.astype(np.uint8))
    pil_gray = pil_gray.resize((size, size), PILImage.BILINEAR)
    gray_resized = np.array(pil_gray, dtype=np.float64) / 255.0

    return gray_resized


def extract_features_giotto(gray_image: np.ndarray) -> np.ndarray:
    """Extract 14 TDA features using giotto-tda."""
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceEntropy, Amplitude, BettiCurve, PersistenceLandscape,
    )

    # Reshape for CubicalPersistence: (n_samples, height, width)
    X = gray_image.reshape(1, *gray_image.shape)

    # Compute persistent homology (H0 and H1)
    cp = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=1)
    diagrams = cp.fit_transform(X)

    features = []

    # 1. Persistence Entropy (H0, H1) → 2 features
    pe = PersistenceEntropy()
    entropy = pe.fit_transform(diagrams)
    features.extend(entropy[0].tolist())

    # 2. Amplitude-Bottleneck (H0, H1) → 2 features
    amp_bottle = Amplitude(metric="bottleneck")
    bottleneck = amp_bottle.fit_transform(diagrams)
    features.extend(bottleneck[0].tolist())

    # 3. Amplitude-Wasserstein (H0, H1) → 2 features
    amp_wass = Amplitude(metric="wasserstein", order=2)
    wasserstein = amp_wass.fit_transform(diagrams)
    features.extend(wasserstein[0].tolist())

    # 4. Betti numbers at 3 filtration thresholds (H0, H1) → 6 features
    # gtda BettiCurve output shape: (n_samples, n_bins, n_homology_dims)
    bc = BettiCurve(n_bins=10)
    betti = bc.fit_transform(diagrams)  # shape: (1, 10, 2)
    # Sample at bins 2, 5, 8 (roughly 0.25, 0.5, 0.75 filtration)
    for dim_idx in range(2):  # H0=0, H1=1
        for threshold_idx in [2, 5, 8]:
            features.append(float(betti[0, threshold_idx, dim_idx]))

    # 5. Persistence Landscape area (H0, H1) → 2 features
    # gtda PersistenceLandscape shape: (n_samples, n_bins, n_homology_dims)
    pl = PersistenceLandscape(n_bins=100, n_layers=1)
    landscape = pl.fit_transform(diagrams)  # shape: (1, 100, 2)
    for dim_idx in range(2):  # H0=0, H1=1
        area = np.sum(np.abs(landscape[0, :, dim_idx]))
        features.append(float(area))

    return np.array(features, dtype=np.float64)


def extract_features_ripser(gray_image: np.ndarray) -> np.ndarray:
    """Extract 14 TDA features using ripser+persim (fallback)."""
    from ripser import ripser as _ripser
    from scipy.spatial.distance import pdist, squareform

    # Create point cloud from image
    coords = []
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i, j] > 0.3:
                coords.append([i, j, gray_image[i, j]])

    if len(coords) < 5:
        return np.zeros(14, dtype=np.float64)

    coords = np.array(coords)

    # Subsample if too many points — deterministic: keep the 200 brightest
    if len(coords) > 200:
        indices = np.argsort(coords[:, 2])[-200:]  # top-200 by intensity
        coords = coords[indices]

    # Run ripser
    result = _ripser(coords, maxdim=1, thresh=2.0)
    diagrams = result["dgms"]

    features = []

    for dim in range(2):  # H0, H1
        if dim < len(diagrams):
            dgm = diagrams[dim]
            # Remove infinite points
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                features.extend([0.0] * 7)
                continue

            lifetimes = finite[:, 1] - finite[:, 0]
            lifetimes = lifetimes[lifetimes > 0]

            if len(lifetimes) == 0:
                features.extend([0.0] * 7)
                continue

            # Persistence Entropy
            probs = lifetimes / lifetimes.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            features.append(entropy)

            # Amplitude-Bottleneck (max lifetime)
            features.append(np.max(lifetimes))

            # Amplitude-Wasserstein (L2 norm)
            features.append(np.sqrt(np.sum(lifetimes ** 2)))

            # Betti numbers at 3 thresholds
            births = finite[:, 0]
            deaths = finite[:, 1]
            thresholds = [0.25, 0.5, 0.75]
            for t in thresholds:
                t_scaled = t * np.max(deaths) if np.max(deaths) > 0 else 0
                betti = np.sum((births <= t_scaled) & (deaths > t_scaled))
                features.append(float(betti))

            # Landscape area (sum of lifetimes as proxy)
            features.append(np.sum(lifetimes))
        else:
            features.extend([0.0] * 7)

    return np.array(features, dtype=np.float64)


def extract_single_image_features(
    image_path: str, size: int = 32, apply_scaler: bool = True
) -> np.ndarray:
    """Extract TDA features from a single image file."""
    img = Image.open(image_path).convert("RGB")
    gray = preprocess_image_for_tda(img, size=size)

    if _detect_backend() == "giotto":
        feats = extract_features_giotto(gray)
    else:
        feats = extract_features_ripser(gray)

    feats = np.delete(feats, 10)
    if apply_scaler:
        scaler_path = Path("data/features/tda_scaler.npy")
        if scaler_path.exists():
            scaler_info = np.load(scaler_path, allow_pickle=True).item()
            feats = (feats - scaler_info["mean"]) / scaler_info["scale"]
    return feats


def extract_single_array_features(
    image_array: np.ndarray, size: int = 32, apply_scaler: bool = True
) -> np.ndarray:
    """Extract TDA features from a numpy array image."""
    gray = preprocess_image_for_tda(image_array, size=size)

    if _detect_backend() == "giotto":
        feats = extract_features_giotto(gray)
    else:
        feats = extract_features_ripser(gray)

    feats = np.delete(feats, 10)
    if apply_scaler:
        scaler_path = Path("data/features/tda_scaler.npy")
        if scaler_path.exists():
            scaler_info = np.load(scaler_path, allow_pickle=True).item()
            feats = (feats - scaler_info["mean"]) / scaler_info["scale"]
    return feats


def _process_single_image_task(args):
    img_path, class_name, size = args
    try:
        feats = extract_single_image_features(img_path, size=size, apply_scaler=False)
        return (feats, f"{class_name}/{Path(img_path).name}", class_name)
    except Exception as e:
        print(f"  Error processing {Path(img_path).name}: {e}")
        return (np.zeros(13, dtype=np.float64), f"{class_name}/{Path(img_path).name}", class_name)


def extract_all_features(
    data_dir: str = "data/raw/Otoscopic_Data",
    output_file: str = "data/features/tda_features.npy",
    size: int = 32,
    class_names: Optional[list] = None,
) -> Tuple[np.ndarray, list]:
    """Extract TDA features for all images in the dataset."""

    if class_names is None:
        class_names = [
            "Acute Otitis Media",
            "Cerumen Impaction",
            "Chronic Otitis Media",
            "Myringosclerosis",
            "Normal",
        ]

    data_path = Path(data_dir)
    tasks = []

    for class_name in class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue

        images = sorted(list(class_dir.glob("*.jpg")) +
                        list(class_dir.glob("*.jpeg")) +
                        list(class_dir.glob("*.png")))
        
        for img_path in images:
            tasks.append((str(img_path), class_name, size))

    print(f"\nExtracting TDA features for {len(tasks)} images in parallel...")
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(_process_single_image_task, tasks), total=len(tasks), desc="TDA Extraction"))

    all_features = []
    all_filenames = []
    all_labels = []

    for feats, fname, label in results:
        all_features.append(feats)
        all_filenames.append(fname)
        all_labels.append(label)

    features_array = np.array(all_features)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(
        output_file,
        {
            "features": features_array,
            "filenames": all_filenames,
            "labels": all_labels,
            "feature_names": get_feature_names(),
        },
    )

    print(f"\nSaved {features_array.shape[0]} feature vectors "
          f"({features_array.shape[1]} dims) to {output_file}")

    return features_array, all_filenames


def get_feature_names() -> list:
    """Return human-readable names for the 13 TDA features."""
    return [
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
        "Betti_H1_t3",
        "Landscape_H0",
        "Landscape_H1",
    ]


if __name__ == "__main__":
    features, filenames = extract_all_features()
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names: {get_feature_names()}")
