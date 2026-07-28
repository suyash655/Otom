# Evaluation

## Metrics

| Metric | Definition | Why |
|---|---|---|
| **Macro-F1** | Unweighted mean of per-class F1 | Primary metric — avoids Normal-class bias |
| **Accuracy** | Correct / Total | Secondary reference |
| **Per-class Precision/Recall** | Logged every 5 epochs | Tracks minority class recovery |
| **MC Dropout Uncertainty** | Std of 10 stochastic forward passes | Safety flag for low-confidence outputs |

## Baseline Results (Checkpoint: `hybrid_best.pth`)

Results populated after training. Run:
```bash
python -m src.train --eval
```

## Explainability Evaluation

| Method | What it measures |
|---|---|
| Grad-CAM | Last conv layer gradient-weighted activations |
| Grad-CAM++ | Improved localisation for multiple objects |
| Integrated Gradients | Attribution relative to black baseline |
| Guided Backpropagation | Only positive gradient flow |

## Reproducing Results

```bash
# 1. Create dummy checkpoint (no data needed)
python -m src.train --dummy

# 2. Full training (requires dataset in data/samples/Otoscopic_Data)
python -m src.train --train --epochs 50 --lr 1e-3

# 3. Evaluate test set
python -m src.train --eval --checkpoint checkpoints/hybrid_best.pth
```

## Known Limitations

- Dataset size: ~700–900 training images (small for deep learning)
- **Not clinically validated** — research prototype only
- MC Dropout uncertainty is an approximation (Bayesian inference is approximate)
- TDA feature extraction is slow (~2–5 s/image on CPU)
