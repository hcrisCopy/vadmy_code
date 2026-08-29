from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.model import DynamicSparseNeuronExpert, SparseNeuronExpert


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complementary CLS-neuron evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.expert_model, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    dynamic = bool(config.pop("dynamic", False))
    model_class = DynamicSparseNeuronExpert if dynamic else SparseNeuronExpert
    model = model_class(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    rows = []
    frame = pd.read_csv(args.manifest)
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export complementary neuron evidence"):
            target = scores / f"{row.key}.npy"
            if not target.exists():
                hidden = torch.from_numpy(
                    load_hidden_array(str(row.hidden_path))
                ).unsqueeze(0).to(device)
                length = torch.tensor([hidden.shape[1]], device=device)
                probability = torch.sigmoid(model(hidden, length))[0, : int(length.item())]
                np.save(target, probability.cpu().numpy().astype(np.float32))
            rows.append({"key": str(row.key), "expert2_score_path": str(target)})
    pd.DataFrame(rows).to_csv(output / "expert2_scores.csv", index=False)
    print(f"wrote {len(rows)} complementary expert curves", flush=True)


if __name__ == "__main__":
    main()
