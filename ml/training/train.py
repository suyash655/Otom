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
from importlib import util
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# ── Local module loading ─────────────────────────────────────────────────────
here = os.path.dirname(__file__)

def _load(name, rel):
    p = os.path.join(here, rel)
    s = util.spec_from_file_location(name, p)
    m = util.module_from_spec(s); s.loader.exec_module(m); return m

hybrid_mod = _load("hybrid_model", "../models/hybrid_model.py")
data_mod   = _load("dataset",      "../preprocessing/dataset.py")

HybridModel     = hybrid_mod.HybridModel
get_dataloaders = data_mod.get_dataloaders
CLASS_NAMES     = data_mod.CLASS_NAMES
CLASS_WEIGHTS   = data_mod.CLASS_WEIGHTS


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

    avg_loss = total_loss / len(loader)
    acc      = correct / total
    rows, macro_f1 = per_class_metrics(all_labels, all_preds, CLASS_NAMES)
    return avg_loss, acc, macro_f1, rows


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
    data_dir:      str   = None,
    splits_file:   str   = None,
    batch_size:    int   = 16,
    epochs:        int   = 50,
    learning_rate: float = 1e-3,
    warmup_epochs: int   = 5,
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu",
    resume_from:   str   = None,
):
    # Set default paths relative to project root
    project_root = Path(here).parent.parent
    if data_dir is None:
        data_dir = str(project_root / "data" / "raw" / "Otoscopic_Data")
    if splits_file is None:
        splits_file = str(project_root / "data" / "splits.json")
    
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
    )
    print(f"Data — Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = HybridModel(
        num_classes=num_classes,
        tda_feature_dim=13,
        pretrained=True,
        dropout=0.4,
        freeze_backbone=False,
    )
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  -> Resumed weights from {resume_from}")
    model.to(dev)

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_criterion(dev)

    # ── Optimiser: differential LR ────────────────────────────────────────────
    # MEDICAL-SAFETY: backbone LR is 1/100 of head LR (not 1/10).
    # 1/10 was too aggressive and destroyed ImageNet pretrained features
    # in the first few epochs on this small dataset (~700 train images).
    # Evidence: train_acc 0.40 with 1/10 ratio; corrected to 1/100.
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.tda_norm.parameters()) +
                       list(model.classifier.parameters()))

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
        val_loss, val_acc, val_f1, val_rows = evaluate(
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
    test_loss, test_acc, test_f1, test_rows = evaluate(
        model, test_loader, criterion, dev, "Test"
    )
    print(f"Test — loss {test_loss:.4f}  acc {test_acc:.4f}  macro_F1 {test_f1:.4f}")
    print_per_class(test_rows, test_f1, split="Test")

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
        "test_acc":          float(test_acc),
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
    checkpoint:  str = None,
    data_dir:    str = None,
    splits_file: str = None,
    batch_size:  int = 16,
    device:      str = "cpu",
):
    # Set default paths relative to project root
    project_root = Path(here).parent.parent
    if checkpoint is None:
        checkpoint = str(project_root / "checkpoints" / "hybrid_best.pth")
    if data_dir is None:
        data_dir = str(project_root / "data" / "raw" / "Otoscopic_Data")
    if splits_file is None:
        splits_file = str(project_root / "data" / "splits.json")
    
    dev  = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev)

    saved_classes = ckpt.get("class_names", [])
    if saved_classes and saved_classes != CLASS_NAMES:
        print(f"WARNING: checkpoint has {saved_classes}, current config has {CLASS_NAMES}")

    model = HybridModel(num_classes=len(CLASS_NAMES), tda_feature_dim=13, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    _, _, test_loader = get_dataloaders(
        data_dir=data_dir, splits_file=splits_file,
        batch_size=batch_size, num_workers=0,
    )
    criterion = build_criterion(dev)
    test_loss, test_acc, test_f1, test_rows = evaluate(
        model, test_loader, criterion, dev, "Test"
    )
    print(f"\nTest — loss {test_loss:.4f}  acc {test_acc:.4f}  macro_F1 {test_f1:.4f}")
    print_per_class(test_rows, test_f1, split="Test")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train/eval HybridModel.")
    parser.add_argument("--dummy",         action="store_true")
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--eval",          action="store_true")
    parser.add_argument("--checkpoint",    type=str,   default=None,
                        help="Path to checkpoint for eval")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume weights from")
    parser.add_argument("--data-dir",      type=str,   default=None,
                        help="Path to data directory")
    parser.add_argument("--splits-file",   type=str,   default=None,
                        help="Path to splits.json file")
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int,   default=3)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
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
        )
    elif args.eval:
        run_eval(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            splits_file=args.splits_file,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        print("Usage:")
        print("  python -m src.train --dummy")
        print("  python -m src.train --train [--epochs 30] [--lr 1e-3]")
        print("  python -m src.train --eval  [--checkpoint checkpoints/hybrid_best.pth]")


if __name__ == "__main__":
    main()