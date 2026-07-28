import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.models.hybrid_model import HybridModel
from ml.preprocessing.dataset import OtoscopyDataset, CLASS_NAMES


class AblationHybridModel(torch.nn.Module):
    """Wrapper model that masks one branch for ablation experiments."""

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


def collect_predictions(model, dataloader, device):
    preds, labels, filenames, raw_classes = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device)
            label = int(batch["label"].item())
            tda_features = batch.get("tda_features")
            if tda_features is None:
                tda_features = torch.zeros(image.shape[0], 13, device=device)
            else:
                tda_features = tda_features.to(device)

            logits = model(image, tda_features)
            pred = int(logits.argmax(dim=1).item())

            preds.append(pred)
            labels.append(label)
            filenames.append(batch["filename"][0])
            raw_classes.append(batch["raw_class"][0])

    return {
        "preds": np.array(preds, dtype=int),
        "labels": np.array(labels, dtype=int),
        "filenames": filenames,
        "raw_classes": raw_classes,
    }


def summarize_disagreement(baseline, ablation, mode):
    same = baseline["preds"] == ablation["preds"]
    disagreements = np.where(~same)[0]

    details = []
    for idx in disagreements:
        details.append({
            "filename": ablation["filenames"][idx],
            "raw_class": ablation["raw_classes"][idx],
            "true_label": int(ablation["labels"][idx]),
            "true_class": CLASS_NAMES[ablation["labels"][idx]],
            "baseline_pred": int(baseline["preds"][idx]),
            "baseline_class": CLASS_NAMES[baseline["preds"][idx]],
            "ablation_pred": int(ablation["preds"][idx]),
            "ablation_class": CLASS_NAMES[ablation["preds"][idx]],
            "baseline_correct": baseline["preds"][idx] == ablation["labels"][idx],
            "ablation_correct": ablation["preds"][idx] == ablation["labels"][idx],
        })

    n_disagree = len(disagreements)
    disagree_by_true = {name: 0 for name in CLASS_NAMES}
    agree_by_true = {name: 0 for name in CLASS_NAMES}
    baseline_better = 0
    ablation_better = 0

    for idx in disagreements:
        true_name = CLASS_NAMES[ablation["labels"][idx]]
        disagree_by_true[true_name] += 1
        if baseline["preds"][idx] == ablation["labels"][idx]:
            baseline_better += 1
        if ablation["preds"][idx] == ablation["labels"][idx]:
            ablation_better += 1

    for idx in np.where(same)[0]:
        true_name = CLASS_NAMES[ablation["labels"][idx]]
        agree_by_true[true_name] += 1

    return {
        "mode": mode,
        "total_samples": int(len(baseline["labels"])),
        "n_disagree": n_disagree,
        "disagreement_rate": float(n_disagree / len(baseline["labels"])),
        "baseline_better": baseline_better,
        "ablation_better": ablation_better,
        "disagreements_by_true_class": disagree_by_true,
        "agreement_by_true_class": agree_by_true,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mine test-set disagreements between hybrid and ablation models."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hybrid_best.pth"))
    parser.add_argument("--data-dir", type=str, default="data/raw/Otoscopic_Data")
    parser.add_argument("--splits-file", type=str, default="data/splits.json")
    parser.add_argument("--batch-size", type=int, default=1)
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
    baseline_preds = collect_predictions(baseline_model, dataloader, device)

    summary = {
        "checkpoint": str(args.checkpoint),
        "ablation_modes": args.ablation_modes,
        "total_samples": int(len(dataset)),
        "results": {},
    }

    for mode in args.ablation_modes:
        model = load_model(args.checkpoint, mode, device)
        ablation_preds = collect_predictions(model, dataloader, device)
        result = summarize_disagreement(baseline_preds, ablation_preds, mode)

        summary["results"][mode] = {
            "total_disagreements": result["n_disagree"],
            "disagreement_rate": result["disagreement_rate"],
            "baseline_better": result["baseline_better"],
            "ablation_better": result["ablation_better"],
            "disagreements_by_true_class": result["disagreements_by_true_class"],
            "agreement_by_true_class": result["agreement_by_true_class"],
        }

        out_path = args.results_dir / f"disagreement_{mode}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved disagreement results for {mode} to {out_path}")

    summary_path = args.results_dir / "disagreement_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
