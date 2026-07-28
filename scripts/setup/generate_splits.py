"""Generate train/val/test splits for an otoscopic dataset and optionally
extract TDA features using the project's `src/features/tda_extract.py`.

Usage (from repository root):
  python scripts/generate_splits.py --data-dir data/raw/Otoscopic_Data \
      --out data/splits.json --val-frac 0.1 --test-frac 0.1 --extract-tda

The script expects the dataset to be organized as:
  data/raw/Otoscopic_Data/<ClassName>/*.jpg
"""
import argparse
import json
import random
from pathlib import Path
import sys
from collections import defaultdict


def make_splits(data_dir: Path, val_frac: float, test_frac: float, seed: int = 42):
    random.seed(seed)
    classes = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    samples = []
    for cls in classes:
        imgs = [p.name for p in cls.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        imgs = sorted(imgs)
        random.shuffle(imgs)
        n = len(imgs)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = max(1, n - n_val - n_test)
        train = imgs[:n_train]
        val = imgs[n_train:n_train + n_val]
        test = imgs[n_train:n_train + n_val:n_test]

        for x in train:
            samples.append({"image": x, "class": cls.name})
        for x in val:
            samples.append({"image": x, "class": cls.name})
        for x in test:
            samples.append({"image": x, "class": cls.name})

    # Build splits dict grouped by split
    by_split = {"train": [], "val": [], "test": []}
    # Re-run to preserve per-split grouping
    for cls in classes:
        imgs = [p.name for p in cls.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        imgs = sorted(imgs)
        random.shuffle(imgs)
        n = len(imgs)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = max(1, n - n_val - n_test)
        train = imgs[:n_train]
        val = imgs[n_train:n_train + n_val]
        test = imgs[n_train + n_val:n_train + n_val + n_test]

        by_split["train"].extend([{"image": x, "class": cls.name} for x in train])
        by_split["val"].extend([{"image": x, "class": cls.name} for x in val])
        by_split["test"].extend([{"image": x, "class": cls.name} for x in test])

    return by_split


def save_splits(splits: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"splits": splits}, f, indent=2)
    print(f"Wrote splits to {out_path} (train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")


def extract_tda(data_dir: Path, out_file: Path, size: int = 32):
    # Ensure src is importable
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / 'src'
    sys.path.insert(0, str(src_dir))
    from features.tda_extract import extract_all_features

    print("Extracting TDA features (this may take time)...")
    extract_all_features(data_dir=str(data_dir), output_file=str(out_file), size=size)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/splits.json'))
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract-tda', action='store_true')
    parser.add_argument('--tda-out', type=Path, default=Path('data/features/tda_features.npy'))
    parser.add_argument('--tda-size', type=int, default=32)
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Data dir {data_dir} not found. Please unzip dataset into this path.")
        return

    splits = make_splits(data_dir, args.val_frac, args.test_frac, seed=args.seed)
    save_splits(splits, args.out)

    if args.extract_tda:
        extract_tda(data_dir, args.tda_out, size=args.tda_size)


if __name__ == '__main__':
    main()
