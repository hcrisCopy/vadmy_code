from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(100 * fraction)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed stratified training subsets for neuron-discovery sensitivity.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(not 0.0 < value <= 1.0 for value in args.fractions):
        raise ValueError("fractions must be in (0, 1]")

    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "subset_metadata.json"
    if args.resume and metadata_path.exists():
        print(f"reuse {metadata_path}", flush=True)
        return

    frame = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)
    class_orders = {
        int(label): rng.permutation(group.index.to_numpy())
        for label, group in frame.groupby("binary_label", sort=True)
    }
    metadata = {"seed": args.seed, "source_manifest": args.manifest, "subsets": {}}
    for fraction in sorted(set(args.fractions)):
        selected = []
        for indices in class_orders.values():
            count = len(indices) if fraction == 1.0 else max(1, int(round(fraction * len(indices))))
            selected.extend(indices[:count].tolist())
        subset = frame.loc[sorted(selected)].reset_index(drop=True)
        path = output / f"{fraction_tag(fraction)}.csv"
        subset.to_csv(path, index=False)
        metadata["subsets"][fraction_tag(fraction)] = {
            "fraction": fraction,
            "videos": len(subset),
            "normal": int((subset.binary_label == 0).sum()),
            "abnormal": int((subset.binary_label == 1).sum()),
            "path": str(path),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
