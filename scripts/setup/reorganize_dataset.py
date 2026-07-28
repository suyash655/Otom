"""Reorganize the tympanic_membrane_dataset into flat class folders."""
import shutil
from pathlib import Path

data_raw = Path("data/raw")
src = data_raw / "tympanic_membrane_dataset"
dst = data_raw / "Otoscopic_Data"

# Create destination
dst.mkdir(parents=True, exist_ok=True)

# Handle normal images
normal_src = src / "normal"
normal_dst = dst / "Normal"
normal_dst.mkdir(exist_ok=True)
if normal_src.exists():
    for img in list(normal_src.glob("*.png")) + list(normal_src.glob("*.jpg")):
        shutil.copy2(img, normal_dst / img.name)
    print(f"Copied {len(list(normal_dst.glob('*.*')))} Normal images")

# Handle abnormal subclasses
abnormal_src = src / "abnormal"
class_mapping = {
    "aom": "Acute Otitis Media",
    "csom": "Chronic Otitis Media",
    "earwax": "Cerumen Impaction",
    "earVentilationTube": "Ear Ventilation Tube",
    "foreignObjectEar": "Foreign Object Ear",
    "otitisexterna": "Otitis Externa",
    "pseudoMembranes": "Pseudo Membranes",
    "tympanoskleros": "Tympanoskleros",
}

if abnormal_src.exists():
    for sub_folder in abnormal_src.iterdir():
        if sub_folder.is_dir():
            class_name = class_mapping.get(sub_folder.name, sub_folder.name.title())
            class_dst = dst / class_name
            class_dst.mkdir(exist_ok=True)
            count = 0
            for img in list(sub_folder.glob("*.png")) + list(sub_folder.glob("*.jpg")):
                shutil.copy2(img, class_dst / img.name)
                count += 1
            print(f"Copied {count} images to {class_name}")

print(f"\nDataset reorganized to: {dst}")
print(f"Classes: {[p.name for p in sorted(dst.iterdir()) if p.is_dir()]}")
