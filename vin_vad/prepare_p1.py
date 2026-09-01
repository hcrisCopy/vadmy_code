from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vin_vad.data import indices_digest


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output(path: Path) -> None:
    if "vadmy_data" not in path.resolve().parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def prepare_split(source_manifest: str, split: str, output: Path) -> list[dict[str, object]]:
    source = pd.read_csv(source_manifest)
    feature_dir = output / "features" / split
    index_dir = output / "frame_indices" / split
    feature_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in tqdm(source.itertuples(index=False), total=len(source), desc=f"cache final CLS/{split}", unit="video"):
        feature_path = feature_dir / f"{row.key}.npy"
        frame_indices_path = index_dir / f"{row.key}.npy"
        if not feature_path.exists() or not frame_indices_path.exists():
            with np.load(str(row.hidden_path), allow_pickle=False) as archive:
                hidden = np.asarray(archive["hidden"])
                indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            valid = int(row.valid_snippets)
            final_layer = np.asarray(hidden[:valid, -1, :], dtype=np.float16)
            valid_indices = indices[:valid]
            if final_layer.shape != (valid, 768):
                raise ValueError(f"{row.key}: expected {(valid, 768)}, got {final_layer.shape}")
            if indices_digest(valid_indices) != str(row.frame_indices_sha256):
                raise RuntimeError(f"{row.key}: frame indices differ from P0 audit")
            np.save(feature_path, final_layer)
            np.save(frame_indices_path, valid_indices)
        cached_shape = np.load(feature_path, mmap_mode="r", allow_pickle=False).shape
        cached_indices = np.load(frame_indices_path, mmap_mode="r", allow_pickle=False)
        if cached_shape != (int(row.valid_snippets), 768) or len(cached_indices) != int(row.valid_snippets):
            raise RuntimeError(f"{row.key}: incomplete final-layer cache; rerun with --clean")
        rows.append(
            {
                "dataset": str(row.dataset),
                "split": split,
                "key": str(row.key),
                "label": str(row.label),
                "binary_label": int(row.binary_label),
                "feature_path": str(feature_path),
                "frame_indices_path": str(frame_indices_path),
                "valid_snippets": int(row.valid_snippets),
                "evaluation_frames": int(row.evaluation_frames),
                "stride": int(row.stride),
            }
        )
    pd.DataFrame(rows).to_csv(output / f"{split}.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache audited final-layer CLS states for P1.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    ensure_output(output)
    signature = {
        "train_manifest": {"path": args.train_manifest, "sha256": file_sha256(args.train_manifest)},
        "test_manifest": {"path": args.test_manifest, "sha256": file_sha256(args.test_manifest)},
        "definition": "CLIP ViT-B/16 block-12 CLS before ln_post/projection",
    }
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("P1 cache inputs changed; rerun with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    completed = output / "audit.json"
    if completed.exists() and (output / "train.csv").exists() and (output / "test.csv").exists():
        print(f"reusing P1 final-layer cache: {completed}", flush=True)
        print(completed.read_text(encoding="utf-8"), flush=True)
        return
    train_rows = prepare_split(args.train_manifest, "train", output)
    test_rows = prepare_split(args.test_manifest, "test", output)
    audit = {
        "status": "pass",
        "train_videos": len(train_rows),
        "test_videos": len(test_rows),
        "train_snippets": sum(int(row["valid_snippets"]) for row in train_rows),
        "test_snippets": sum(int(row["valid_snippets"]) for row in test_rows),
    }
    completed.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
