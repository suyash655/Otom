import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.models.hybrid_model import HybridModel
from ml.preprocessing.dataset import OtoscopyDataset, CLASS_NAMES


class AblationHybridModel(torch.nn.Module):
    def __init__(self, base_model: HybridModel, mode: str = "hybrid"):
        super().__init__()
        if mode not in {"hybrid", "cnn_only", "tda_only"}:
            raise ValueError(f"Unsupported mode: {mode}")
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


def build_model(ablation_mode: str, num_classes: int, tda_feature_dim: int = 13):
    base_model = HybridModel(
        num_classes=num_classes,
        tda_feature_dim=tda_feature_dim,
        pretrained=False,
        dropout=0.0,
        freeze_backbone=False,
    )
    if ablation_mode == "hybrid":
        return base_model
    return AblationHybridModel(base_model, mode=ablation_mode)


def load_model(checkpoint: Path, ablation_mode: str, device: torch.device):
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model = build_model(ablation_mode, num_classes=len(CLASS_NAMES))
    if isinstance(model, AblationHybridModel):
        model.model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device).eval()
    return model


def class_counts(labels, class_names):
    counts = {name: 0 for name in class_names}
    for label in labels:
        counts[class_names[int(label)]] += 1
    return counts


def compare_confusion_matrices(base_cm, ablation_cm):
    assert base_cm.shape == ablation_cm.shape
    delta = ablation_cm.astype(int) - base_cm.astype(int)
    return delta.tolist()


def evaluate_confusion_matrix(model, dataloader, device):
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            tda_features = batch.get("tda_features")
            if tda_features is None:
                tda_features = torch.zeros(image.shape[0], 13, device=device)
            else:
                tda_features = tda_features.to(device)

            logits = model(image, tda_features)
            preds = logits.argmax(dim=1)
            for t, p in zip(label.cpu().tolist(), preds.cpu().tolist()):
                cm[t, p] += 1

    return cm


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-class confusion matrix deltas for ablation models."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hybrid_best.pth"))
    parser.add_argument("--data-dir", type=str, default="data/raw/Otoscopic_Data")
    parser.add_argument("--splits-file", type=str, default="data/splits.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ablation-modes", nargs="+", default=["cnn_only", "tda_only"])
    parser.add_argument("--results-dir", type=Path, default=Path("results/ablation"))
    args = parser.parse_args()

    device = torch.device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = OtoscopyDataset(
        data_dir=args.data_dir,
        splits_file=args.splits_file,
        split="test",
        tda_features_file="data/features/tda_features_clean.npy",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline_model = load_model(args.checkpoint, "hybrid", device)
    baseline_cm = evaluate_confusion_matrix(baseline_model, dataloader, device)

    summary = {
        "checkpoint": str(args.checkpoint),
        "ablation_modes": args.ablation_modes,
        "confusion_matrix_delta": {},
    }

    for mode in args.ablation_modes:
        model = load_model(args.checkpoint, mode, device)
        ablation_cm = evaluate_confusion_matrix(model, dataloader, device)
        delta = compare_confusion_matrices(baseline_cm, ablation_cm)

        result = {
            "mode": mode,
            "baseline_confusion_matrix": baseline_cm.tolist(),
            "ablation_confusion_matrix": ablation_cm.tolist(),
            "delta_confusion_matrix": delta,
        }

        out_path = args.results_dir / f"per_class_delta_{mode}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved per-class delta for {mode} to {out_path}")

        summary["confusion_matrix_delta"][mode] = delta

    summary_path = args.results_dir / "per_class_delta_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
