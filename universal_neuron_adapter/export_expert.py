from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.model import SparseNeuronExpert


def load_model(path: str, device: torch.device) -> SparseNeuronExpert:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = SparseNeuronExpert(config["active_per_layer"], config["temporal_width"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reusable snippet curves from the sparse neuron expert.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out_dir)
    score_dir = output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.expert_model, device)
    rows = []
    frame = pd.read_csv(args.manifest)
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export neuron evidence"):
            target = score_dir / f"{row.key}.npy"
            hidden = torch.from_numpy(load_hidden_array(str(row.hidden_path))).unsqueeze(0).to(device)
            length = torch.tensor([hidden.shape[1]], device=device)
            probability = torch.sigmoid(model(hidden, length))[0, : int(length.item())].cpu().numpy().astype(np.float32)
            np.save(target, probability)
            rows.append({
                "key": str(row.key), "label": str(row.label), "binary_label": int(row.binary_label),
                "expert_score_path": str(target), "snippets": len(probability),
            })
    pd.DataFrame(rows).to_csv(output / "expert_scores.csv", index=False)
    print(f"wrote {len(rows)} expert curves to {output}", flush=True)


if __name__ == "__main__":
    main()

