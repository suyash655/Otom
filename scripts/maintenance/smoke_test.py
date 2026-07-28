"""Quick smoke test — run from project root: python scripts/smoke_test.py"""
import sys; sys.path.insert(0, '.')
import torch
from src.data.dataset import CLASS_NAMES, CLASS_WEIGHTS, FOLDER_TO_CLASS, OtoscopyDataset
from src.train import build_criterion
from src.train import CLASS_NAMES as TN, CLASS_WEIGHTS as TW

print("=== Class config ===")
for i, (n, w) in enumerate(zip(TN, TW)):
    print(f"  [{i}] {n:<25} weight={w}")

print()
print("=== Weighted criterion ===")
crit = build_criterion(torch.device('cpu'))
print("  weight tensor:", crit.weight.tolist())

print()
print("=== Dataset smoke test ===")
ds = OtoscopyDataset('data/raw/Otoscopic_Data', 'data/splits.json', 'train')
sample = ds[0]
print(f"  sample[0] -> class_name={sample['class_name']} raw_class={sample['raw_class']} label={sample['label'].item()}")

from collections import Counter
label_counts = Counter()
for s in ds.samples:
    from src.data.dataset import FOLDER_TO_CLASS as F2C
    label_counts[F2C.get(s['class'], '???')] += 1
print("  Label distribution in train split:")
for cls, n in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"    {cls:<25} {n}")

other_raw = [s['class'] for s in ds.samples if F2C.get(s['class']) == 'Other']
print(f"\n  Other-mapped samples: {len(other_raw)}")
print(f"  Source folders in Other: {sorted(set(other_raw))}")
print()
print("=== All checks passed ===")
