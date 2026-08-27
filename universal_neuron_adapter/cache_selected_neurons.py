from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache selected CLS neuron coordinates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    score_dir = output / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["neurons"]
    layers = np.asarray([row["layer"] - 1 for row in selection], dtype=np.int64)
    dimensions = np.asarray([row["dimension"] for row in selection], dtype=np.int64)
    rows = []
    frame = pd.read_csv(args.manifest)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="cache selected CLS neurons"):
        target = score_dir / f"{row.key}.npy"
        if not target.exists():
            hidden = np.load(str(row.hidden_path))["hidden"]
            np.save(target, hidden[:, layers, dimensions].astype(np.float16))
        rows.append({
            "key": str(row.key), "label": str(row.label),
            "binary_label": int(row.binary_label), "selected_path": str(target),
        })
    pd.DataFrame(rows).to_csv(output / "selected_manifest.csv", index=False)
    print(f"cached {len(rows)} videos with {len(selection)} selected neurons", flush=True)


if __name__ == "__main__":
    main()
