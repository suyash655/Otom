import numpy as np

tda = np.load("data/features/tda_features.npy", allow_pickle=True).item()
features = tda["features"]
filenames = tda["filenames"]

print(f"Total TDA vectors: {features.shape}")
print(f"Feature dim: {features.shape[1]}")

print("\n--- Zero rows (missing/failed extractions) ---")
zero_rows = np.where(np.all(features == 0, axis=1))[0]
print(f"Zero vectors: {len(zero_rows)} / {len(features)} ({100*len(zero_rows)/len(features):.1f}%)")

print("\n--- Feature value ranges ---")
names = ["PersEntropy_H0","PersEntropy_H1","AmpBottleneck_H0","AmpBottleneck_H1",
         "AmpWasserstein_H0","AmpWasserstein_H1","Betti_H0_t1","Betti_H0_t2",
         "Betti_H0_t3","Betti_H1_t1","Betti_H1_t2","Betti_H1_t3",
         "Landscape_H0","Landscape_H1"]
for i, name in enumerate(names):
    col = features[:, i]
    print(f"  {name:<25} min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}  std={col.std():.4f}")

print("\n--- NaN / Inf check ---")
print(f"NaNs: {np.isnan(features).sum()}")
print(f"Infs: {np.isinf(features).sum()}")

print("\n--- Sample filenames (first 5) ---")
for f in filenames[:5]:
    print(f"  {f}")