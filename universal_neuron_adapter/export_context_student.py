from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from universal_neuron_adapter.context_student import context_student_scores, load_context_student
from universal_neuron_adapter.normality import load_normality_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multi-scale directional CLS-neuron scores.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--normality-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    scores = output / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    student = load_context_student(args.student_model)
    normality = load_normality_model(args.normality_model)
    rows = []
    frame = pd.read_csv(args.manifest)
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="export context student"):
        target = scores / f"{row.key}.npy"
        if not target.exists():
            np.save(target, context_student_scores(str(row.hidden_path), normality, student))
        rows.append({"key": str(row.key), "student_score_path": str(target)})
    pd.DataFrame(rows).to_csv(output / "student_scores.csv", index=False)
    print(f"wrote {len(rows)} context student curves", flush=True)


if __name__ == "__main__":
    main()
