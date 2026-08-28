from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.category_neurons import category_evidence, load_category_model
from universal_neuron_adapter.data import load_hidden_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Export category-conditional CLS-neuron evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    model = load_category_model(args.expert_model)
    rows = []
    frame = pd.read_csv(args.manifest)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export category neuron evidence"):
        target = scores / f"{row.key}.npy"
        if not target.exists():
            np.save(target, category_evidence(load_hidden_array(str(row.hidden_path)), model))
        rows.append({"key": str(row.key), "expert2_score_path": str(target)})
    pd.DataFrame(rows).to_csv(output / "expert2_scores.csv", index=False)
    print(f"wrote {len(rows)} category expert curves", flush=True)


if __name__ == "__main__":
    main()
