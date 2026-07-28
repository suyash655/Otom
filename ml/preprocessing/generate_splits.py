import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Use paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw" / "Otoscopic_Data"
    
    # We will use the same folders as FOLDER_TO_CLASS mapping in dataset.py
    folders = [
        "Acute Otitis Media",
        "Cerumen Impaction",
        "Chronic Otitis Media",
        "Myringosclerosis",
        "Normal",
        "Otitis Externa",
        "Tympanoskleros",
        "Ear Ventilation Tube",
        "Pseudo Membranes",
        "Foreign Object Ear"
    ]
    
    all_files = []
    
    for folder in folders:
        d = data_dir / folder
        if d.exists():
            files = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            for f in files:
                all_files.append({"image": f.name, "class": folder})
                
    print(f"Total images found: {len(all_files)}")
    
    # Stratify by folder
    y = [item["class"] for item in all_files]
    
    # 80 / 10 / 10 split
    # Handle rare classes with < 3 samples (can't stratify well)
    # Actually, let's just do a simple split or handle stratification safely
    train_val, test, y_train_val, y_test = train_test_split(all_files, y, test_size=0.1, stratify=y, random_state=42)
    train, val, y_train, y_val = train_test_split(train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42) # 0.1111 * 0.9 = ~0.1
    
    splits = {
        "splits": {
            "train": train,
            "val": val,
            "test": test
        }
    }
    
    splits_file = project_root / "data" / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_file} (Train: {len(train)}, Val: {len(val)}, Test: {len(test)})")
    
    # Now, fit scaler on train set and apply to all
    features_dir = project_root / "data" / "features"
    tda_file = features_dir / "tda_features.npy"
    
    if tda_file.exists():
        tda_data = np.load(tda_file, allow_pickle=True).item()
        features = tda_data["features"]
        filenames = tda_data["filenames"]
        
        # Identify train indices
        train_keys = set([f"{item['class']}/{item['image']}" for item in train])
        train_indices = [i for i, fn in enumerate(filenames) if fn in train_keys]
        
        train_feats = features[train_indices]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        
        # Save scaler
        scaler_file = features_dir / "tda_scaler.npy"
        np.save(scaler_file, {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "feature_names": tda_data["feature_names"]
        })
        print(f"Fit StandardScaler on {len(train_feats)} train samples and saved to {scaler_file}")
        
        # Apply scaler to all features
        features_clean = scaler.transform(features)
        tda_data["features"] = features_clean
        
        clean_file = features_dir / "tda_features_clean.npy"
        np.save(clean_file, tda_data)
        print(f"Saved scaled features to {clean_file}")
    else:
        print(f"Warning: {tda_file} not found. Skipping scaler creation.")

if __name__ == "__main__":
    main()
