from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from universal_neuron_adapter.data import resample_curve
from universal_neuron_adapter.model import ScoreCorrectionHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate corrected scores with strict author-style frame alignment.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--expert-manifest", required=True)
    parser.add_argument("--correction-model", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--baseline", choices=["dsanet", "desc", "lagovad"], required=True)
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.correction_model, map_location="cpu")
    if checkpoint.get("baseline") != args.baseline or checkpoint.get("dataset") != args.dataset:
        raise ValueError("correction checkpoint baseline/dataset mismatch")
    model = ScoreCorrectionHead(checkpoint["config"]["width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    baseline = pd.read_csv(args.baseline_manifest)
    expert = pd.read_csv(args.expert_manifest)[["key", "expert_score_path"]]
    frame = baseline.merge(expert, on="key", validate="one_to_one")
    baseline_curves, corrected_curves, rows = [], [], []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"evaluate {args.baseline}/{args.dataset}"):
            base = np.load(str(row.baseline_score_path)).astype(np.float32)
            neuron = resample_curve(np.load(str(row.expert_score_path)), len(base))
            base_tensor = torch.from_numpy(base).unsqueeze(0).to(device)
            neuron_tensor = torch.from_numpy(neuron).unsqueeze(0).to(device)
            lengths = torch.tensor([len(base)], device=device)
            corrected = torch.sigmoid(model(base_tensor, neuron_tensor, lengths))[0].cpu().numpy().astype(np.float32)
            baseline_curves.append(base)
            corrected_curves.append(corrected)
            rows.append({
                "key": str(row.key), "snippets": len(base),
                "baseline_mean": float(base.mean()), "expert_mean": float(neuron.mean()),
                "corrected_mean": float(corrected.mean()),
            })
    truth = np.load(args.gt_path).astype(np.int64).reshape(-1)
    baseline_frames = np.repeat(np.concatenate(baseline_curves), args.frames_per_snippet)
    corrected_frames = np.repeat(np.concatenate(corrected_curves), args.frames_per_snippet)
    if len(truth) != len(baseline_frames) or len(truth) != len(corrected_frames):
        raise RuntimeError(
            f"strict frame alignment failed: gt={len(truth)} baseline={len(baseline_frames)} corrected={len(corrected_frames)}"
        )
    metrics = {
        "baseline": {
            "auc": float(roc_auc_score(truth, baseline_frames)),
            "ap": float(average_precision_score(truth, baseline_frames)),
        },
        "corrected": {
            "auc": float(roc_auc_score(truth, corrected_frames)),
            "ap": float(average_precision_score(truth, corrected_frames)),
        },
        "frames": len(truth),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "per_video.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

