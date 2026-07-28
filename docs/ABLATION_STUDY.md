# Ablation Study

This document describes the ablation evaluation workflow for the Hybrid CNN+TDA model.

## Goals

- Compare the full `hybrid` architecture against `cnn_only` and `tda_only` variants.
- Capture per-class performance and confusion matrix deltas.
- Mine test-set disagreements between hybrid and ablation predictions.
- Export structured JSON results for downstream analysis and reporting.

## New tooling

### `scripts/setup/train.py`

Added support for:

- `--ablation-mode {hybrid,cnn_only,tda_only}`
- `--results-dir results/ablation`
- JSON export of ablation evaluation metrics with per-class results and confusion matrix

This allows running both training and eval with a controlled ablation branch.

### `analysis/disagreement_mining.py`

Purpose:

- Load a saved checkpoint.
- Run the hybrid model and one or more ablation variants over the test set.
- Record examples where predictions differ.
- Track whether disagreements improve or harm test-set accuracy.

Outputs:

- `results/ablation/disagreement_{mode}.json`
- `results/ablation/disagreement_summary.json`

### `analysis/per_class_delta.py`

Purpose:

- Compare confusion matrices between the baseline hybrid model and ablation variants.
- Report per-class deltas in false positives and false negatives.

Outputs:

- `results/ablation/per_class_delta_{mode}.json`
- `results/ablation/per_class_delta_summary.json`

## Recommended usage

1. Train or evaluate with a hybrid checkpoint:

```bash
python scripts/setup/train.py --train --ablation-mode hybrid --results-dir results/ablation
python scripts/setup/train.py --eval --checkpoint checkpoints/hybrid_best.pth --ablation-mode cnn_only --results-dir results/ablation
```

2. Mine disagreements:

```bash
python analysis/disagreement_mining.py --checkpoint checkpoints/hybrid_best.pth --device cpu
```

3. Compute per-class confusion matrix deltas:

```bash
python analysis/per_class_delta.py --checkpoint checkpoints/hybrid_best.pth --device cpu
```

## Notes

- The `cnn_only` ablation zeroes the TDA branch before classification.
- The `tda_only` ablation zeroes the CNN branch before classification.
- Results are saved as JSON so they can be loaded into notebooks, dashboards, or reports.

## Next steps

- Add plots for per-class delta matrices and disagreement trends.
- Combine ablation results with explanation heatmaps for qualitative validation.
- Document clinical implications of TDA vs. CNN contributions.
