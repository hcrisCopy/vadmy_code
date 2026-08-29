from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.data import load_hidden_array
from universal_neuron_adapter.dispersion import dispersion_evidence, load_dispersion_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export temporal-dispersion neuron evidence.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expert-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    model = load_dispersion_model(args.expert_model)
    rows = []
    frame = pd.read_csv(args.manifest)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export dispersion evidence"):
        target = scores / f"{row.key}.npy"
        if not target.exists():
            evidence = dispersion_evidence(load_hidden_array(str(row.hidden_path)), model)
            np.save(target, evidence)
        rows.append({"key": str(row.key), "expert2_score_path": str(target)})
    pd.DataFrame(rows).to_csv(output / "expert2_scores.csv", index=False)
    print(f"wrote {len(rows)} dispersion expert curves", flush=True)


if __name__ == "__main__":
    main()
