import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load original
tda = np.load("data/features/tda_features.npy", allow_pickle=True).item()
features  = tda["features"]     # (857, 14)
filenames = tda["filenames"]
labels    = tda["labels"]

# Drop Betti_H1_t2 (index 11 — zero variance, no signal)
# MEDICAL-SAFETY: confirm index before dropping
names = ["PersEntropy_H0","PersEntropy_H1","AmpBottleneck_H0","AmpBottleneck_H1",
         "AmpWasserstein_H0","AmpWasserstein_H1","Betti_H0_t1","Betti_H0_t2",
         "Betti_H0_t3","Betti_H1_t1","Betti_H1_t2","Betti_H1_t3",
         "Landscape_H0","Landscape_H1"]

dead_idx = 10  # Betti_H1_t2 — std=0 confirmed
print(f"Dropping feature {dead_idx}: {names[dead_idx]}")

keep_idx = [i for i in range(14) if i != dead_idx]
features_clean = features[:, keep_idx]
names_clean    = [names[i] for i in keep_idx]

print(f"Shape before: {features.shape} → after: {features_clean.shape}")

# Standardize: zero mean, unit variance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_clean)

print("\nAfter standardization:")
for i, n in enumerate(names_clean):
    print(f"  {n:<25} mean={features_scaled[:,i].mean():.4f}  std={features_scaled[:,i].std():.4f}")

# Save scaler params so inference can apply same transform
Path("data/features").mkdir(parents=True, exist_ok=True)
np.save("data/features/tda_scaler.npy", {
    "mean": scaler.mean_,
    "scale": scaler.scale_,
    "feature_names": names_clean,
    "dropped_idx": dead_idx,
    "dropped_name": names[dead_idx],
})

# Save clean features
np.save("data/features/tda_features_clean.npy", {
    "features":      features_scaled,
    "filenames":     filenames,
    "labels":        labels,
    "feature_names": names_clean,
})

print(f"\nSaved clean features to data/features/tda_features_clean.npy")
print(f"Saved scaler to data/features/tda_scaler.npy")
print(f"New feature dim: {features_scaled.shape[1]} (was 14)")