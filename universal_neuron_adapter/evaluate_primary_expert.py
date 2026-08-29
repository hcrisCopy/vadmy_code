from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from universal_neuron_adapter.data import HiddenVideoDataset, collate_hidden
from universal_neuron_adapter.model import SparseNeuronExpert, topk_bag


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a sparse CLS-neuron expert at video level.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_file)
    if args.resume and output.exists():
        print(f"reuse {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.expert_model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SparseNeuronExpert(int(config["active_per_layer"]), int(config["temporal_width"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        HiddenVideoDataset(args.manifest, args.maximum_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_hidden,
    )
    labels, scores, keys = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate primary expert"):
            lengths = batch["lengths"].to(device)
            logits = model(batch["hidden"].to(device), lengths)
            scores.extend(topk_bag(torch.sigmoid(logits), lengths).cpu().tolist())
            labels.extend(batch["labels"].tolist())
            keys.extend(batch["keys"])
    metrics = {
        "video_auc": float(roc_auc_score(labels, scores)),
        "video_ap": float(average_precision_score(labels, scores)),
        "videos": len(labels),
        "manifest": args.manifest,
        "checkpoint": args.expert_model,
        "test_usage": "post-hoc reporting only" if "test" in Path(args.manifest).name else "training-derived validation",
    }
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"key": keys, "binary_label": labels, "score": scores}).to_csv(
        output.with_suffix(".csv"), index=False
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
