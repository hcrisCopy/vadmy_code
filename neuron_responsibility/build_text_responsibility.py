#!/usr/bin/env python3
"""Select text-grounded CLIP neurons and build aligned responsibility priors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.common import (
    clean_output,
    grouped_rows,
    is_normal_label,
    load_hidden,
    read_feature_csv,
    read_hidden_manifest,
    resample_feature,
    save_json,
    uniform_indices,
    write_csv,
)
from neuron_responsibility.text_responsibility import (
    NORMAL_CONCEPTS,
    TextMargin,
    abnormal_concepts,
    calibrated_prior,
    encode_concepts,
    evidence_from_responsibility,
    load_reference_clip,
    robust_location_scale,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hidden_for(mapping: dict[str, str], key: str, layer: int) -> np.ndarray:
    hidden, _ = load_hidden(mapping[key])
    if not 0 <= layer < hidden.shape[1]:
        raise IndexError(f"{key}: layer index {layer} outside {hidden.shape}")
    return hidden[:, layer, :]


def normal_statistics(
    groups: dict[str, pd.DataFrame], mapping: dict[str, str], dataset: str,
    layer: int, samples_per_video: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(768, dtype=np.float64)
    square = np.zeros(768, dtype=np.float64)
    count = 0
    for key, group in tqdm(groups.items(), desc="normal neuron statistics", unit="video"):
        if key not in mapping or not is_normal_label(dataset, str(group.iloc[0]["label"])):
            continue
        hidden = hidden_for(mapping, key, layer)
        hidden = hidden[uniform_indices(len(hidden), samples_per_video)].astype(np.float64)
        total += hidden.sum(axis=0)
        square += np.square(hidden).sum(axis=0)
        count += len(hidden)
    if count == 0:
        raise RuntimeError("no matched normal hidden states")
    center = total / count
    variance = np.maximum(square / count - np.square(center), 1e-12)
    return center.astype(np.float32), np.sqrt(variance).astype(np.float32), count


def video_responsibilities(
    groups: dict[str, pd.DataFrame], mapping: dict[str, str], dataset: str,
    layer: int, samples_per_video: int, tail_fraction: float,
    margin: TextMargin, center: torch.Tensor, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    summaries, labels, keys = [], [], []
    for key, group in tqdm(groups.items(), desc="text neuron responsibility", unit="video"):
        if key not in mapping:
            continue
        hidden = hidden_for(mapping, key, layer)
        hidden = hidden[uniform_indices(len(hidden), samples_per_video)]
        tensor = torch.from_numpy(hidden).to(device)
        _, responsibility = margin.responsibility(tensor, center)
        tail = max(1, int(np.ceil(len(tensor) * tail_fraction)))
        summaries.append(torch.topk(responsibility, tail, dim=0).values.mean(0).cpu().numpy())
        labels.append(not is_normal_label(dataset, str(group.iloc[0]["label"])))
        keys.append(key)
    return np.stack(summaries), np.asarray(labels, dtype=bool), keys


def stable_selection(
    summaries: np.ndarray, abnormal: np.ndarray, keys: list[str], topk: int, folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = summaries[abnormal].mean(0) - summaries[~abnormal].mean(0)
    fold_scores = []
    buckets = np.asarray([sum(bytearray(key.encode("utf-8"))) % folds for key in keys])
    for fold in range(folds):
        keep = buckets != fold
        fold_scores.append(
            summaries[keep & abnormal].mean(0) - summaries[keep & ~abnormal].mean(0)
        )
    fold_scores = np.stack(fold_scores)
    stability = (fold_scores > 0).mean(0)
    robust_score = np.maximum(score, 0.0) * stability
    selected = np.argsort(-robust_score)[:topk]
    return selected.astype(np.int64), score.astype(np.float32), stability.astype(np.float32)


def raw_evidence(
    hidden: np.ndarray, indices: np.ndarray, margin: TextMargin,
    center: torch.Tensor, scale: torch.Tensor, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(hidden.astype(np.float32)).to(device)
    _, responsibility = margin.responsibility(tensor, center)
    semantic, structural = evidence_from_responsibility(
        responsibility, tensor, center, scale,
        torch.as_tensor(indices, dtype=torch.long, device=device),
    )
    return semantic.cpu().numpy(), structural.cpu().numpy()


def fit_calibration(
    groups, mapping, dataset, layer, selected, random_indices,
    margin, center, scale, samples_per_video, device,
) -> tuple[dict[str, float], dict[str, float]]:
    selected_values = [[], []]
    random_values = [[], []]
    for key, group in tqdm(groups.items(), desc="normal evidence calibration", unit="video"):
        if key not in mapping or not is_normal_label(dataset, str(group.iloc[0]["label"])):
            continue
        hidden = hidden_for(mapping, key, layer)
        hidden = hidden[uniform_indices(len(hidden), samples_per_video)]
        for target, indices in ((selected_values, selected), (random_values, random_indices)):
            semantic, structural = raw_evidence(hidden, indices, margin, center, scale, device)
            target[0].append(semantic)
            target[1].append(structural)

    def finish(values):
        semantic = np.concatenate(values[0])
        structural = np.concatenate(values[1])
        semantic_center, semantic_scale = robust_location_scale(semantic)
        structural_center, structural_scale = robust_location_scale(structural)
        return {
            "semantic_center": semantic_center, "semantic_scale": semantic_scale,
            "structural_center": structural_center, "structural_scale": structural_scale,
        }
    return finish(selected_values), finish(random_values)


def build_split(
    name: str, source_csv: str, manifest: str, output: Path, dataset: str, layer: int,
    selected: np.ndarray, random_indices: np.ndarray, selected_calibration: dict,
    random_calibration: dict, margin: TextMargin, center: torch.Tensor,
    scale: torch.Tensor, device: torch.device, clean: bool,
) -> dict[str, int]:
    frame = read_feature_csv(source_csv)
    mapping, _ = read_hidden_manifest(manifest)
    feature_dir = output / name / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows, missing, reused = [], set(), 0
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"build {name} priors", unit="clip"):
        clip_path = str(row["path"])
        from neuron_responsibility.common import base_key
        key = base_key(clip_path)
        if key not in mapping:
            missing.add(key)
            continue
        target = feature_dir / f"{Path(clip_path).stem}.npy"
        if target.exists() and not clean:
            reused += 1
        else:
            clip = np.load(clip_path, mmap_mode="r")
            hidden = resample_feature(hidden_for(mapping, key, layer), len(clip))
            sem, struct = raw_evidence(hidden, selected, margin, center, scale, device)
            r_sem, r_struct = raw_evidence(hidden, random_indices, margin, center, scale, device)
            prior = calibrated_prior(sem, struct, selected_calibration)
            random_prior = calibrated_prior(r_sem, r_struct, random_calibration)
            np.save(target, np.stack([prior, random_prior, sem, struct], axis=1).astype(np.float32))
        rows.append([clip_path, str(target), str(row["label"]), key])
    csv_path = output / f"{name}.csv"
    write_csv(csv_path, ["clip_path", "neuron_path", "label", "key"], rows)
    return {"rows": len(rows), "missing_videos": len(missing), "reused": reused}


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-grounded CLIP neuron responsibility builder.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--clip-root", required=True)
    parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--snippets-per-video", type=int, default=256)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--projection-videos", type=int, default=32)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0 < args.tail_fraction <= 1 or not 1 <= args.topk <= 768:
        parser.error("require 0 < tail-fraction <= 1 and 1 <= topk <= 768")
    seed_everything(args.seed)
    output = clean_output(args.out_dir, args.clean)
    artifact_path = output / "selected_text_neurons.json"
    if artifact_path.exists() and not args.clean:
        raise RuntimeError("selection exists; use --clean or choose another --out-dir")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenize = load_reference_clip(args.clip_root, args.clip_weight, device)
    normal_text = encode_concepts(model, tokenize, NORMAL_CONCEPTS, device)
    abnormal_names = abnormal_concepts(args.dataset)
    abnormal_text = encode_concepts(model, tokenize, abnormal_names, device)
    margin = TextMargin(model, normal_text, abnormal_text)
    train_frame = read_feature_csv(args.train_list)
    groups = grouped_rows(train_frame)
    mapping, _ = read_hidden_manifest(args.train_hidden_manifest)
    center_np, scale_np, normal_count = normal_statistics(
        groups, mapping, args.dataset, args.layer, args.snippets_per_video
    )
    center = torch.from_numpy(center_np).to(device)
    scale = torch.from_numpy(scale_np).to(device).clamp_min(1e-6)
    summaries, labels, keys = video_responsibilities(
        groups, mapping, args.dataset, args.layer, args.snippets_per_video,
        args.tail_fraction, margin, center, device,
    )
    selected, score, stability = stable_selection(summaries, labels, keys, args.topk, args.folds)
    candidates = np.setdiff1d(np.arange(768), selected)
    random_indices = np.random.default_rng(args.seed).choice(candidates, args.topk, replace=False)
    selected_calibration, random_calibration = fit_calibration(
        groups, mapping, args.dataset, args.layer, selected, random_indices,
        margin, center, scale, args.snippets_per_video, device,
    )
    concept_profiles = []
    probe_hidden = (center + scale).unsqueeze(0)
    for concept_index in range(len(abnormal_names)):
        concept_margin = TextMargin(model, normal_text, abnormal_text[concept_index:concept_index + 1])
        _, contribution = concept_margin.responsibility(probe_hidden, center)
        concept_profiles.append(contribution[0, selected].abs().cpu().numpy())
    concept_profiles = np.stack(concept_profiles)
    concept_probability = concept_profiles / np.maximum(concept_profiles.sum(0, keepdims=True), 1e-12)
    normalized_entropy = -(
        concept_probability * np.log(np.maximum(concept_probability, 1e-12))
    ).sum(0) / np.log(max(2, len(abnormal_names)))

    projection_cosines = []
    for key, group in tqdm(list(groups.items())[:args.projection_videos], desc="validate CLIP projection", unit="video"):
        if key not in mapping:
            continue
        clip = np.load(str(group.iloc[0]["path"])).astype(np.float32)
        hidden = resample_feature(hidden_for(mapping, key, args.layer), len(clip))
        with torch.no_grad():
            projected = margin.image(torch.from_numpy(hidden).to(device))
            target = F.normalize(torch.from_numpy(clip).to(device), dim=-1)
            projection_cosines.extend((projected * target).sum(-1).cpu().tolist())
    projection = {
        "mean_cosine": float(np.mean(projection_cosines)),
        "p05_cosine": float(np.quantile(projection_cosines, 0.05)),
        "samples": len(projection_cosines),
    }
    artifact = {
        "method": "clip_text_neuron_responsibility_v1",
        "dataset": args.dataset, "layer_zero_based": args.layer,
        "selected_indices": selected.tolist(), "random_control_indices": random_indices.tolist(),
        "selection_score": score.tolist(), "fold_stability": stability.tolist(),
        "normal_center": center_np.tolist(), "normal_scale": scale_np.tolist(),
        "normal_stat_snippets": normal_count,
        "normal_concepts": list(NORMAL_CONCEPTS), "abnormal_concepts": abnormal_names,
        "selected_concept_entropy_mean": float(normalized_entropy.mean()),
        "selected_concept_entropy": normalized_entropy.tolist(),
        "selected_calibration": selected_calibration, "random_calibration": random_calibration,
        "projection_validation": projection,
        "selection_source": "video labels plus frozen CLIP text margin; no baseline anomaly score",
    }
    save_json(artifact_path, artifact)
    result = {
        "selection": str(artifact_path), "projection_validation": projection,
        "train": build_split(
            "train", args.train_list, args.train_hidden_manifest, output, args.dataset,
            args.layer, selected, random_indices, selected_calibration, random_calibration,
            margin, center, scale, device, args.clean,
        ),
        "test": build_split(
            "test", args.test_list, args.test_hidden_manifest, output, args.dataset,
            args.layer, selected, random_indices, selected_calibration, random_calibration,
            margin, center, scale, device, args.clean,
        ),
    }
    (output / "build_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
