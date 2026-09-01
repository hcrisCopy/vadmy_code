from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vin_vad.data import base_key, indices_digest, is_normal_label
from vin_vad.evaluate import expand_snippet_scores


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_array_header(path: str, array_name: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read an array shape/dtype from NPZ without decompressing its payload."""
    member = f"{array_name}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"{path}: missing array {array_name}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            shape, _, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(value) for value in shape), np.dtype(dtype)


def ensure_output_path(path: Path) -> None:
    resolved = path.resolve()
    if "vadmy_data" not in resolved.parts:
        raise ValueError("out-dir must be inside the sibling vadmy_data directory")


def clean_output(path: Path) -> None:
    ensure_output_path(path)
    if path.exists():
        shutil.rmtree(path)


def feature_rows(path: str) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError(f"{path}: DSANet CSV must contain path and label")
    frame["key"] = frame["path"].map(base_key)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby("key", sort=False):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError(f"{key}: inconsistent labels {sorted(labels)}")
        crop_lengths: list[int] = []
        for feature_path in group["path"].astype(str):
            feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            if feature.ndim != 2:
                raise ValueError(f"{feature_path}: expected [T,D], got {feature.shape}")
            crop_lengths.append(int(feature.shape[0]))
        # DSANet's __0..__9 files are spatial crops of one timeline, not
        # consecutive temporal chunks. Their temporal lengths must agree and
        # are counted once when defining the evaluator domain.
        if len(set(crop_lengths)) != 1:
            raise ValueError(f"{key}: DSANet crop lengths differ: {crop_lengths}")
        rows.append({"key": str(key), "label": next(iter(labels)), "feature_snippets": crop_lengths[0]})
    return rows


def read_partial(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_split(
    dataset: str,
    split: str,
    dsanet_csv: str,
    hidden_manifest: str,
    frames_per_snippet: int,
    output: Path,
) -> list[dict[str, object]]:
    expected = feature_rows(dsanet_csv)
    expected_by_key = {str(row["key"]): row for row in expected}
    hidden = pd.read_csv(hidden_manifest)
    required = {"key", "hidden_path", "stride", "num_frames", "layers"}
    missing = required - set(hidden.columns)
    if missing:
        raise ValueError(f"{hidden_manifest}: missing columns {sorted(missing)}")
    hidden["key"] = hidden["key"].map(base_key)
    if hidden["key"].duplicated().any():
        raise ValueError(f"{hidden_manifest}: duplicate video keys")
    hidden_by_key = hidden.set_index("key", drop=False)
    missing_hidden = [row["key"] for row in expected if row["key"] not in hidden_by_key.index]
    extra_hidden = [key for key in hidden_by_key.index if key not in expected_by_key]
    if missing_hidden or extra_hidden:
        raise RuntimeError(
            f"{split}: hidden/DSANet keys differ; missing={missing_hidden[:5]}, extra={extra_hidden[:5]}"
        )

    partial = output / f"{split}.partial.jsonl"
    rows = read_partial(partial)
    completed = {str(row["key"]) for row in rows}
    mode = "a" if rows else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for feature_row in tqdm(expected, desc=f"P0 {dataset}/{split}", unit="video"):
            key = str(feature_row["key"])
            if key in completed:
                continue
            manifest_row = hidden_by_key.loc[key]
            hidden_path = str(manifest_row.hidden_path)
            hidden_shape, hidden_dtype = npz_array_header(hidden_path, "hidden")
            with np.load(hidden_path, allow_pickle=False) as archive:
                required_arrays = {"hidden", "frame_indices", "num_frames", "stride", "layers"}
                absent = required_arrays - set(archive.files)
                if absent:
                    raise ValueError(f"{hidden_path}: missing arrays {sorted(absent)}")
                frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                num_frames = int(archive["num_frames"])
                stride = int(archive["stride"])
                layers = np.asarray(archive["layers"], dtype=np.int64)
            if len(hidden_shape) != 3 or hidden_shape[1:] != (12, 768):
                raise ValueError(f"{hidden_path}: expected [T,12,768], got {hidden_shape}")
            if hidden_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
                raise ValueError(f"{hidden_path}: expected float16/float32 hidden, got {hidden_dtype}")
            raw_hidden_snippets = hidden_shape[0]
            if len(frame_indices) != raw_hidden_snippets:
                raise ValueError(f"{hidden_path}: hidden/frame_indices length mismatch")
            if stride != frames_per_snippet or int(manifest_row.stride) != frames_per_snippet:
                raise ValueError(f"{key}: expected stride {frames_per_snippet}, got {stride}")
            if num_frames != int(manifest_row.num_frames):
                raise ValueError(f"{key}: manifest/NPZ num_frames mismatch")
            if not np.array_equal(layers, np.arange(1, 13, dtype=np.int64)):
                raise ValueError(f"{key}: expected CLIP layers 1..12, got {layers.tolist()}")
            expected_indices = np.arange(0, num_frames, stride, dtype=np.int64)
            if not np.array_equal(frame_indices, expected_indices):
                raise ValueError(f"{key}: frame_indices do not match stride-{stride} extraction")

            valid_snippets = int(feature_row["feature_snippets"])
            floor_snippets = num_frames // frames_per_snippet
            ceil_snippets = (num_frames + frames_per_snippet - 1) // frames_per_snippet
            if split == "test" and valid_snippets != floor_snippets:
                raise ValueError(
                    f"{key}: DSANet test has {valid_snippets} snippets but its GT domain requires "
                    f"floor(num_frames/stride)={floor_snippets}"
                )
            if split == "train" and valid_snippets not in {floor_snippets, ceil_snippets}:
                raise ValueError(
                    f"{key}: DSANet train has {valid_snippets} snippets; expected floor/ceil "
                    f"count {floor_snippets}/{ceil_snippets}"
                )
            if valid_snippets > raw_hidden_snippets:
                raise ValueError(f"{key}: DSANet needs more snippets than hidden states contain")
            valid_indices = frame_indices[:valid_snippets]
            result = {
                "dataset": dataset,
                "split": split,
                "key": key,
                "label": str(feature_row["label"]),
                "binary_label": int(not is_normal_label(dataset, str(feature_row["label"]))),
                "hidden_path": hidden_path,
                "raw_hidden_snippets": int(raw_hidden_snippets),
                "valid_snippets": valid_snippets,
                "dropped_tail_snippets": int(raw_hidden_snippets - valid_snippets),
                "raw_num_frames": num_frames,
                "evaluation_frames": min(num_frames, valid_snippets * frames_per_snippet),
                "stride": stride,
                "first_frame_index": int(valid_indices[0]),
                "last_frame_index": int(valid_indices[-1]),
                "frame_indices_sha256": indices_digest(valid_indices),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(result)
    if len(rows) != len(expected):
        raise RuntimeError(f"{split}: audited {len(rows)}/{len(expected)} videos")
    pd.DataFrame(rows).to_csv(output / f"{split}.csv", index=False)
    partial.unlink()
    return rows


def save_alignment_probe(test_rows: list[dict[str, object]], gt: np.ndarray, output: Path) -> dict[str, object]:
    offset = 0
    chosen: tuple[dict[str, object], np.ndarray, int] | None = None
    for row in test_rows:
        length = int(row["evaluation_frames"])
        gt_slice = gt[offset : offset + length]
        if int(row["binary_label"]) == 1 and np.any(gt_slice > 0):
            first_positive = int(np.flatnonzero(gt_slice > 0)[0])
            chosen = (row, gt_slice, first_positive)
            break
        offset += length
    if chosen is None:
        raise RuntimeError("test GT has no positive frame for the alignment probe")
    row, gt_slice, first_positive = chosen
    with np.load(str(row["hidden_path"]), allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)[: int(row["valid_snippets"])]
    snippet_index = min(first_positive // int(row["stride"]), len(frame_indices) - 1)
    snippet_curve = np.zeros(len(frame_indices), dtype=np.float32)
    snippet_curve[snippet_index] = 1.0
    frame_curve = expand_snippet_scores(snippet_curve, frame_indices, int(row["evaluation_frames"]))
    active = np.flatnonzero(frame_curve > 0)
    probe_path = output / "alignment_probe.npz"
    np.savez_compressed(
        probe_path,
        key=np.asarray(str(row["key"])),
        snippet_index=np.asarray(snippet_index),
        snippet_curve=snippet_curve,
        frame_curve=frame_curve,
        frame_indices=frame_indices,
        gt_slice=gt_slice.astype(np.float32),
    )
    return {
        "key": str(row["key"]),
        "snippet_index": snippet_index,
        "expanded_start": int(active[0]),
        "expanded_end_exclusive": int(active[-1] + 1),
        "first_positive_gt_frame": first_positive,
        "artifact": str(probe_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ViN-VAD hidden states against the official DSANet evaluator domain.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output = Path(args.out_dir)
    ensure_output_path(output)
    if args.clean:
        clean_output(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "dataset": args.dataset,
        "inputs": {
            name: {"path": value, "sha256": file_sha256(value)}
            for name, value in (
                ("train_csv", args.train_csv),
                ("test_csv", args.test_csv),
                ("train_hidden_manifest", args.train_hidden_manifest),
                ("test_hidden_manifest", args.test_hidden_manifest),
                ("gt_path", args.gt_path),
            )
        },
        "frames_per_snippet": args.frames_per_snippet,
    }
    signature_path = output / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("P0 inputs changed; rerun with --clean")
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    audit_path = output / "audit.json"
    if audit_path.exists() and (output / "train.csv").exists() and (output / "test.csv").exists():
        print(f"reusing completed P0 audit: {audit_path}", flush=True)
        print(audit_path.read_text(encoding="utf-8"), flush=True)
        return

    train_rows = audit_split(
        args.dataset, "train", args.train_csv, args.train_hidden_manifest, args.frames_per_snippet, output
    )
    test_rows = audit_split(
        args.dataset, "test", args.test_csv, args.test_hidden_manifest, args.frames_per_snippet, output
    )
    train_keys = {str(row["key"]) for row in train_rows}
    test_keys = {str(row["key"]) for row in test_rows}
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"official train/test overlap: {sorted(overlap)[:5]}")
    gt = np.asarray(np.load(args.gt_path, allow_pickle=False), dtype=np.float32).reshape(-1)
    evaluation_frames = sum(int(row["evaluation_frames"]) for row in test_rows)
    if evaluation_frames != len(gt):
        raise RuntimeError(f"test frame total {evaluation_frames} != GT length {len(gt)}")
    probe = save_alignment_probe(test_rows, gt, output)
    audit = {
        "status": "pass",
        "dataset": args.dataset,
        "protocol": "unchanged DSANet frame-level evaluator domain",
        "frames_per_snippet": args.frames_per_snippet,
        "train_videos": len(train_rows),
        "test_videos": len(test_rows),
        "train_test_key_overlap": 0,
        "test_gt_frames": len(gt),
        "test_evaluation_frames": evaluation_frames,
        "train_tail_snippets_dropped": sum(int(row["dropped_tail_snippets"]) for row in train_rows),
        "test_tail_snippets_dropped": sum(int(row["dropped_tail_snippets"]) for row in test_rows),
        "alignment_probe": probe,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
