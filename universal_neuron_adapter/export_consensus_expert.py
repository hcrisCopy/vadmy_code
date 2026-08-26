from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden
from universal_neuron_adapter.model import ConsensusNeuronExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out_dir)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = ConsensusNeuronExpert(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        frame = pd.read_csv(args.manifest)
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export consensus neuron expert"):
            hidden = torch.from_numpy(load_hidden(str(row.hidden_path))).unsqueeze(0).to(device)
            lengths = torch.tensor([hidden.shape[1]], device=device)
            probability = torch.sigmoid(model(hidden, lengths))[0].cpu().numpy().astype(np.float32)
            path = scores / f"{row.key}.npy"
            np.save(path, probability)
            rows.append({"key": str(row.key), "expert_score_path": str(path)})
    pd.DataFrame(rows).to_csv(output / "expert_scores.csv", index=False)


if __name__ == "__main__":
    main()
