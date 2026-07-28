"""
Training / inference script for the HybridModel.

Provides:
  --dummy  : Run dummy forward pass and save checkpoint
  --train  : Full training loop with weighted loss, per-class metrics, checkpointing
  --eval   : Evaluate on test set from a saved checkpoint
"""
import os
import glob
import shutil
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# Import from canonical ML module
from ml.models.hybrid_model import HybridModel
from ml.preprocessing.dataset import get_dataloaders, CLASS_NAMES, CLASS_WEIGHTS


# ── Weighted loss ─────────────────────────────────────────────────────────────

def build_criterion(device: torch.device) -> nn.CrossEntropyLoss:
    """CrossEntropyLoss with class weights from dataset config.

    MEDICAL-SAFETY: weights are defined in dataset.py next to class definitions
    so they stay in sync. Never hardcode weights here independently.
    """
    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


# ── Scheduler: linear warmup -> cosine annealing ──────────────────────────────

def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 3):
    """3-epoch linear warmup then cosine annealing to eta_min=1e-6.

    Warmup prevents the randomly-initialised classifier head from producing
    large gradients that corrupt pretrained backbone weights in epoch 1.

    MEDICAL-SAFETY: warmup_epochs=3 is tuned for this dataset size (~700 train).
    Increase to 5 if dataset grows beyond 2000 images.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear ramp 0->1
        return 1.0  # cosine scheduler takes over after this

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler


class AblationHybridModel(torch.nn.Module):
    """Wrapper model that masks one branch for ablation experiments."""

    def __init__(self, base_model: HybridModel, mode: str = "hybrid"):
        super().__init__()
        if mode not in {"hybrid", "cnn_only", "tda_only"}:
            raise ValueError(f"Unsupported ablation_mode: {mode}")
        self.mode = mode
        self.model = base_model

    def forward(self, images: torch.Tensor, tda_features: torch.Tensor) -> torch.Tensor:
        cnn_feats, tda_feats, _, _ = self.model.get_all_features(images, tda_features)
        if self.mode == "cnn_only":
            tda_feats = torch.zeros_like(tda_feats)
        elif self.mode == "tda_only":
            cnn_feats = torch.zeros_like(cnn_feats)
        combined = torch.cat([cnn_feats, tda_feats], dim=1)
        return self.model.classifier(combined)

    @torch.no_grad()
    def predict_with_uncertainty(self, images, tda_features, n_forward: int = 10):
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        predictions = []
        for _ in range(n_forward):
            logits = self.forward(images, tda_features)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)

        self.eval()
        stacked = torch.stack(predictions)
        mean_probs = stacked.mean(dim=0)
        uncertainty = stacked.std(dim=0).mean(dim=1)
        return mean_probs, uncertainty, stacked


def build_model(
    ablation_mode: str,
    num_classes: int,
    tda_feature_dim: int = 13,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
):
    base_model = HybridModel(
        num_classes=num_classes,
        tda_feature_dim=tda_feature_dim,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    if ablation_mode == "hybrid":
        return base_model
    return AblationHybridModel(base_model, mode=ablation_mode)


# ── Per-class metrics ─────────────────────────────────────────────────────────

def per_class_metrics(all_labels, all_preds, class_names):
    """Compute per-class precision, recall, F1 and macro averages."""
    n = len(class_names)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for true, pred in zip(all_labels, all_preds):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true]  += 1

    rows = []
    f1s  = []
    for i, name in enumerate(class_names):
        p  = tp[i] / (tp[i] + fp[i] + 1e-8)
        r  = tp[i] / (tp[i] + fn[i] + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        support = tp[i] + fn[i]
        rows.append({"class": name, "precision": p, "recall": r,
                     "f1": f1, "support": support})
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return rows, macro_f1


def print_per_class(rows, macro_f1, split="Val"):
    print(f"\n  {split} per-class metrics:")
    print(f"  {'Class':<25} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-'*52}")
    for r in rows:
        print(f"  {r['class']:<25} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['support']:>5}")
    print(f"  {'Macro F1':<25} {'':>6} {'':>6} {macro_f1:>6.3f}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        images    = batch["image"].to(device)
        labels    = batch["label"].to(device)
        tda_feats = batch.get("tda_features")
        if tda_feats is None:
            tda_feats = torch.zeros(images.shape[0], 14, device=device)
        else:
            tda_feats = tda_feats.to(device)

        optimizer.zero_grad()
        logits = model(images, tda_feats)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients on small medical dataset
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
            images    = batch["image"].to(device)
            labels    = batch["label"].to(device)
            tda_feats = batch.get("tda_features")
            if tda_feats is None:
                tda_feats = torch.zeros(images.shape[0], 13, device=device)  # was 14, dropped dead feature
            else:
                tda_feats = tda_feats.to(device)

            logits = model(images, tda_feats)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cm[t, p] += 1

    avg_loss = total_loss / len(loader)
    acc      = correct / total
    rows, macro_f1 = per_class_metrics(all_labels, all_preds, CLASS_NAMES)
    return avg_loss, acc, macro_f1, rows, cm


# ── Dummy check ───────────────────────────────────────────────────────────────

def run_dummy_check(batch_size: int = 2, device: str = "cpu"):
    dev = torch.device(device)
    model = HybridModel(num_classes=len(CLASS_NAMES), tda_feature_dim=13, pretrained=False)
    model.to(dev).eval()

    images = torch.randn(batch_size, 3, 224, 224, device=dev)
    tda    = torch.randn(batch_size, 13, device=dev)

    with torch.no_grad():
        logits = model(images, tda)

    print(f"Dummy forward pass OK: output shape {logits.shape}")
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "hybrid_dummy.pth"
    torch.save({"model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


# ── Full training ─────────────────────────────────────────────────────────────

def run_training(
    data_dir:      str   = "data/raw/Otoscopic_Data",
    splits_file:   str   = "data/splits.json",
    batch_size:    int   = 16,
    epochs:        int   = 50,
    learning_rate: float = 1e-3,
    warmup_epochs: int   = 5,
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu",
    resume_from:   str   = None,
    ablation_mode: str   = "hybrid",
    results_dir:   str   = "results/ablation",
):
    dev         = torch.device(device)
    num_classes = len(CLASS_NAMES)

    print(f"\n{'='*62}")
    print(f"  HybridModel training — {num_classes} classes")
    for i, (n, w) in enumerate(zip(CLASS_NAMES, CLASS_WEIGHTS)):
        print(f"    [{i}] {n:<25}  weight={w:.4f}")
    print(f"  Device: {dev} | Epochs: {epochs} | Head LR: {learning_rate}")
    print(f"  Backbone LR: {learning_rate/100:.2e}  (1/100 of head LR)")
    print(f"  Warmup: {warmup_epochs} epochs -> cosine annealing")
    print(f"{'='*62}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=data_dir,
    splits_file=splits_file,
    batch_size=batch_size,
    num_workers=0,
    tda_features_file="data/features/tda_features_clean.npy",  # add this line
)
    print(f"Data — Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        ablation_mode=ablation_mode,
        num_classes=num_classes,
        tda_feature_dim=13,
        pretrained=True,
        dropout=0.4,
        freeze_backbone=False,
    )
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=dev)
        if isinstance(model, AblationHybridModel):
            model.model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        print(f"  -> Resumed weights from {resume_from}")
    model.to(dev)

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_criterion(dev)

    # ── Optimiser: differential LR ────────────────────────────────────────────
    # MEDICAL-SAFETY: backbone LR is 1/100 of head LR (not 1/10).
    # 1/10 was too aggressive and destroyed ImageNet pretrained features
    # in the first few epochs on this small dataset (~700 train images).
    base_model = model.model if isinstance(model, AblationHybridModel) else model
    backbone_params = list(base_model.backbone.parameters())
    head_params     = (list(base_model.tda_norm.parameters()) +
                       list(base_model.classifier.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": learning_rate / 100},   # MEDICAL-SAFETY: 1/100
        {"params": head_params,     "lr": learning_rate},
    ], weight_decay=1e-4)

    # ── Scheduler: warmup + cosine ────────────────────────────────────────────
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_macro_f1 = 0.0
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    history  = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, dev
        )
        val_loss, val_acc, val_f1, val_rows, val_cm = evaluate(
            model, val_loader, criterion, dev, "Val"
        )
        scheduler.step()

        # Log current LRs so we can verify warmup is working
        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr     = optimizer.param_groups[1]["lr"]

        print(f"Epoch {epoch+1:>2}/{epochs} | "
              f"loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} "
              f"val_F1 {val_f1:.4f} | "
              f"lr_head {head_lr:.2e} lr_bb {backbone_lr:.2e}")

        # Save best by macro-F1 (not accuracy — avoids Normal-bias)
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1

            # Priority 4: versioned checkpoint filenames
            versioned_name = f"hybrid_best_v{epoch+1}_{val_f1:.3f}.pth"
            versioned_path = ckpt_dir / versioned_name

            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "class_names": CLASS_NAMES,
                "num_classes": num_classes,
            }
            torch.save(ckpt_data, versioned_path)

            # Keep hybrid_best.pth updated for API compatibility
            best_path = ckpt_dir / "hybrid_best.pth"
            shutil.copy2(str(versioned_path), str(best_path))

            # Prune: keep only the last 3 versioned checkpoints
            versioned_ckpts = sorted(
                glob.glob(str(ckpt_dir / "hybrid_best_v*.pth")),
                key=os.path.getmtime,
            )
            for old_ckpt in versioned_ckpts[:-3]:
                os.remove(old_ckpt)
                print(f"  -> Pruned old checkpoint: {Path(old_ckpt).name}")

            print(f"  -> Saved best (macro_F1={val_f1:.4f}) as {versioned_name}")

        # Print per-class every 5 epochs so we can track minority class recovery
        if (epoch + 1) % 5 == 0:
            print_per_class(val_rows, val_f1, split=f"Epoch {epoch+1} Val")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss), "train_acc": float(train_acc),
            "val_loss": float(val_loss),     "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr_head": float(head_lr),       "lr_backbone": float(backbone_lr),
        })

    print_per_class(val_rows, val_f1, split="Final Val")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_rows, test_cm = evaluate(
        model, test_loader, criterion, dev, "Test"
    )
    print(f"Test — loss {test_loss:.4f}  acc {test_acc:.4f}  macro_F1 {test_f1:.4f}")
    print_per_class(test_rows, test_f1, split="Test")

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        ablation_name = ablation_mode or "hybrid"
        ablation_metrics = {
            "ablation_mode":    ablation_name,
            "class_names":      CLASS_NAMES,
            "class_weights":    CLASS_WEIGHTS,
            "train_loss":       float(train_loss),
            "train_acc":        float(train_acc),
            "val_loss":         float(val_loss),
            "val_acc":          float(val_acc),
            "val_macro_f1":     float(val_f1),
            "test_loss":        float(test_loss),
            "test_acc":         float(test_acc),
            "test_macro_f1":    float(test_f1),
            "best_val_macro_f1":float(best_macro_f1),
            "per_class_test":   test_rows,
            "confusion_matrix": test_cm.tolist(),
        }
        out_ablation = results_dir / f"{ablation_name}_metrics.json"
        with open(out_ablation, "w") as f:
            json.dump(ablation_metrics, f, indent=2)
        print(f"\nSaved ablation metrics to {out_ablation}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "class_names":       CLASS_NAMES,
        "class_weights":     CLASS_WEIGHTS,
        "train_loss":        float(train_loss),
        "train_acc":         float(train_acc),
        "val_loss":          float(val_loss),
        "val_acc":           float(val_acc),
        "val_macro_f1":      float(val_f1),
        "test_loss":         float(test_loss),
        "test_acc":         float(test_acc),
        "test_macro_f1":     float(test_f1),
        "best_val_macro_f1": float(best_macro_f1),
        "per_class_test":    test_rows,
        "history":           history,
    }
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to checkpoints/metrics.json")


# ── Eval-only ─────────────────────────────────────────────────────────────────

def run_eval(
    checkpoint:  str = "checkpoints/hybrid_best.pth",
    data_dir:    str = "data/raw/Otoscopic_Data",
    splits_file: str = "data/splits.json",
    batch_size:  int = 16,
    device:      str = "cpu",
    ablation_mode: str = "hybrid",
    results_dir: str = "results/ablation",
):
    dev  = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev)

    saved_classes = ckpt.get("class_names", [])
    if saved_classes and saved_classes != CLASS_NAMES:
        print(f"WARNING: checkpoint has {saved_classes}, current config has {CLASS_NAMES}")

    model = build_model(
        ablation_mode=ablation_mode,
        num_classes=len(CLASS_NAMES),
        tda_feature_dim=13,
        pretrained=False,
        dropout=0.0,
        freeze_backbone=False,
    )
    if isinstance(model, AblationHybridModel):
        model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    _, _, test_loader = get_dataloaders(
        data_dir=data_dir, splits_file=splits_file,
        batch_size=batch_size, num_workers=0,
    )
    criterion = build_criterion(dev)
    test_loss, test_acc, test_f1, test_rows, test_cm = evaluate(
        model, test_loader, criterion, dev, "Test"
    )
    print(f"\nTest — loss {test_loss:.4f}  acc {test_acc:.4f}  macro_F1 {test_f1:.4f}")
    print_per_class(test_rows, test_f1, split="Test")

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        metrics = {
            "ablation_mode": ablation_mode,
            "class_names":      CLASS_NAMES,
            "class_weights":    CLASS_WEIGHTS,
            "test_loss":        float(test_loss),
            "test_acc":         float(test_acc),
            "test_macro_f1":    float(test_f1),
            "per_class_test":   test_rows,
            "confusion_matrix": test_cm.tolist(),
        }
        out_path = Path(results_dir) / f"{ablation_mode}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train/eval HybridModel.")
    parser.add_argument("--dummy",         action="store_true")
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--eval",          action="store_true")
    parser.add_argument("--checkpoint",    type=str,   default="checkpoints/hybrid_best.pth")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume weights from")
    parser.add_argument("--data-dir",      type=str,   default="data/raw/Otoscopic_Data")
    parser.add_argument("--splits-file",   type=str,   default="data/splits.json")
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int,   default=3)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ablation-mode", choices=["hybrid", "cnn_only", "tda_only"],
                        default="hybrid",
                        help="Run the model in ablation mode.")
    parser.add_argument("--results-dir", type=str, default="results/ablation",
                        help="Directory to save ablation evaluation JSON results.")
    args = parser.parse_args()

    if args.dummy:
        run_dummy_check(device=args.device)
    elif args.train:
        run_training(
            data_dir=args.data_dir,
            splits_file=args.splits_file,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            resume_from=args.resume,
            ablation_mode=args.ablation_mode,
            results_dir=args.results_dir,
        )
    elif args.eval:
        run_eval(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            splits_file=args.splits_file,
            batch_size=args.batch_size,
            device=args.device,
            ablation_mode=args.ablation_mode,
            results_dir=args.results_dir,
        )
    else:
        print("Usage:")
        print("  python -m src.train --dummy")
        print("  python -m src.train --train [--epochs 30] [--lr 1e-3]")
        print("  python -m src.train --eval  [--checkpoint checkpoints/hybrid_best.pth]")


if __name__ == "__main__":
    main()
