from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.model import SparseNeuronExpert


def main() -> None:
    parser = argparse.ArgumentParser(description="Export primary CLS-neuron evidence curves.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control", choices=["none", "remove_selected", "random_matched"], default="none")
    parser.add_argument("--control-seed", type=int, default=3407)
    args = parser.parse_args()
    output = Path(args.out_dir)
    score_dir = output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.expert_model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SparseNeuronExpert(config["active_per_layer"], config["temporal_width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    selected = [indices.cpu().numpy() for indices in model.gates().topk(model.active_per_layer, dim=-1).indices]
    rng = np.random.default_rng(args.control_seed)
    controlled = []
    for indices in selected:
        if args.control == "remove_selected":
            controlled.append(indices)
        elif args.control == "random_matched":
            candidates = np.setdiff1d(np.arange(768), indices)
            controlled.append(rng.choice(candidates, size=len(indices), replace=False))
        else:
            controlled.append(np.asarray([], dtype=np.int64))
    rows = []
    frame = pd.read_csv(args.manifest)
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export primary neuron expert"):
            hidden = torch.from_numpy(load_hidden_array(str(row.hidden_path))).unsqueeze(0).to(device)
            for layer, dimensions in enumerate(controlled):
                if len(dimensions):
                    neutral = hidden[:, :, layer, :].mean(dim=-1, keepdim=True)
                    hidden[:, :, layer, dimensions] = neutral
            length = torch.tensor([hidden.shape[1]], device=device)
            score = torch.sigmoid(model(hidden, length))[0, : int(length.item())].cpu().numpy().astype(np.float32)
            target = score_dir / f"{row.key}.npy"
            np.save(target, score)
            rows.append({"key": str(row.key), "label": str(row.label), "binary_label": int(row.binary_label), "expert_score_path": str(target), "snippets": len(score)})
    pd.DataFrame(rows).to_csv(output / "expert_scores.csv", index=False)
    print(f"wrote {len(rows)} expert curves to {output}", flush=True)


if __name__ == "__main__":
    main()
