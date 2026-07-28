# Data Model

## Class Labels

| Index | Class Name | Description | Weight |
|---|---|---|---|
| 0 | Normal | Healthy tympanic membrane | 1.0 |
| 1 | Acute Otitis Media | Acute bacterial/viral middle ear infection | ~2.5 |
| 2 | Otitis Media with Effusion | Fluid in middle ear, no acute infection | ~2.5 |
| 3 | Chronic Suppurative Otitis Media | Chronic ear discharge + perforation | ~3.0 |
| 4 | Myringosclerosis | Calcium deposits on tympanic membrane | ~3.5 |

> Weights are stored in `backend/app/services/data_ingestion.py → CLASS_WEIGHTS`.

## Input Specification

| Field | Value |
|---|---|
| Image format | PNG / JPEG / BMP |
| Image size | Any (resized to 224×224 internally) |
| Colour space | RGB |
| Normalisation | ImageNet mean/std |

## TDA Feature Vector

Dimension: **13** (after removing 1 dead feature from original 14)

| Index | Feature | Description |
|---|---|---|
| 0–3 | H0 statistics | Connected component persistence (birth, death, mean, std) |
| 4–7 | H1 statistics | Loop persistence (birth, death, mean, std) |
| 8–9 | H0 count | Number of connected components (total, persistent) |
| 10–11 | H1 count | Number of loops (total, persistent) |
| 12 | Betti ratio | H1/H0 ratio |

Features stored in: `data/schemas/tda_features_clean.npy`  
Scaler stored in: `data/schemas/tda_scaler.npy`

## Dataset Splits

Stored in `data/schemas/splits.json`:

```json
{
  "train": ["path/to/img1.jpg", ...],
  "val":   ["path/to/img2.jpg", ...],
  "test":  ["path/to/img3.jpg", ...]
}
```

## Checkpoint Format

```python
{
  "model_state_dict": {...},   # PyTorch state dict
  "epoch": int,
  "val_acc": float,
  "val_macro_f1": float,
  "class_names": [str, ...],
  "num_classes": int
}
```
