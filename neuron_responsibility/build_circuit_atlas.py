#!/usr/bin/env python3
"""Discover class-specific CLIP circuits and build compact aligned features."""

from __future__ import annotations

import argparse
import hashlib
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

from neuron_responsibility.baselines import label_map_for
from neuron_responsibility.common import (
    base_key,
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
    encode_concepts,
    load_reference_clip,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_names_for(dataset: str) -> list[str]:
    values = list(label_map_for(dataset).values())
    normal = "Normal" if dataset == "ucf" else "normal"
    return [value for value in values if value != normal]


def label_classes(dataset: str, label: str, class_names: list[str]) -> list[int]:
    mapping = label_map_for(dataset)
    raw = [label] if dataset == "ucf" else str(label).split("-")
    return [class_names.index(mapping[value]) for value in raw if value in mapping and mapping[value] in class_names]


def hidden_layer(mapping: dict[str, str], key: str, layer: int) -> np.ndarray:
    hidden, _ = load_hidden(mapping[key])
    if not 0 <= layer < hidden.shape[1]:
        raise IndexError(f"{key}: layer {layer} is outside hidden shape {hidden.shape}")
    return hidden[:, layer, :]


def collect_statistics(
    groups: dict[str, pd.DataFrame],
    mapping: dict[str, str],
    dataset: str,
    layer: int,
    snippets_per_video: int,
    tail_fraction: float,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    total = np.zeros(768, dtype=np.float64)
    square = np.zeros(768, dtype=np.float64)
    normal_count = 0
    matched: list[tuple[str, str, np.ndarray]] = []
    missing: list[str] = []
    for key, group in tqdm(groups.items(), desc="load circuit statistics", unit="video"):
        if key not in mapping:
            missing.append(key)
            continue
        label = str(group.iloc[0]["label"])
        hidden = hidden_layer(mapping, key, layer)
        sample = hidden[uniform_indices(len(hidden), snippets_per_video)].astype(np.float32)
        matched.append((key, label, sample))
        if is_normal_label(dataset, label):
            total += sample.astype(np.float64).sum(axis=0)
            square += np.square(sample.astype(np.float64)).sum(axis=0)
            normal_count += len(sample)
    if normal_count == 0:
        raise RuntimeError("no normal hidden states matched the manifest")
    center = total / normal_count
    variance = np.maximum(square / normal_count - np.square(center), 1e-12)
    scale = np.sqrt(variance)
    normal_positive, normal_negative = [], []
    class_positive: list[list[np.ndarray]] = [[] for _ in class_names]
    class_negative: list[list[np.ndarray]] = [[] for _ in class_names]
    tail = max(1, int(np.ceil(snippets_per_video * tail_fraction)))
    for _, label, sample in tqdm(matched, desc="summarize class tails", unit="video"):
        z = (sample - center) / np.maximum(scale, 1e-6)
        count = min(tail, len(z))
        positive = np.partition(z, len(z) - count, axis=0)[-count:].mean(axis=0)
        negative = -np.partition(z, count - 1, axis=0)[:count].mean(axis=0)
        if is_normal_label(dataset, label):
            normal_positive.append(positive)
            normal_negative.append(negative)
        else:
            for index in label_classes(dataset, label, class_names):
                class_positive[index].append(positive)
                class_negative[index].append(negative)
    normal_pos = np.mean(normal_positive, axis=0)
    normal_neg = np.mean(normal_negative, axis=0)
    positive_excess, negative_excess = [], []
    for index, name in enumerate(class_names):
        if not class_positive[index]:
            raise RuntimeError(f"no matched training video for class {name}")
        positive_excess.append(np.mean(class_positive[index], axis=0) - normal_pos)
        negative_excess.append(np.mean(class_negative[index], axis=0) - normal_neg)
    report = {
        "matched_videos": len(matched),
        "normal_stat_snippets": normal_count,
        "missing_videos": len(set(missing)),
        "class_video_counts": {
            name: len(class_positive[index]) for index, name in enumerate(class_names)
        },
    }
    return (
        center.astype(np.float32),
        scale.astype(np.float32),
        np.stack(positive_excess).astype(np.float32),
        np.stack(negative_excess).astype(np.float32),
        np.asarray(matched, dtype=object),
        report,
    )


def concept_gradients(
    model,
    normal_text: torch.Tensor,
    abnormal_text: torch.Tensor,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    work = torch.from_numpy(center + 0.10 * scale).to(device).unsqueeze(0)
    center_tensor = torch.from_numpy(center).to(device)
    values = []
    for index in tqdm(range(len(abnormal_text)), desc="concept gradient profiles", unit="class"):
        margin = TextMargin(model, normal_text, abnormal_text[index:index + 1])
        _, contribution = margin.responsibility(work, center_tensor)
        values.append((contribution[0] / torch.from_numpy(0.10 * scale).to(device).clamp_min(1e-6)).cpu().numpy())
    return np.stack(values).astype(np.float32)


def select_circuits(
    positive_excess: np.ndarray,
    negative_excess: np.ndarray,
    gradients: np.ndarray,
    topk_per_class: int,
    specificity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    direction = np.where(positive_excess >= negative_excess, 1.0, -1.0).astype(np.float32)
    activation = np.maximum(positive_excess, negative_excess).clip(min=0.0)
    semantic = np.maximum(direction * gradients, 0.0)
    scores = np.zeros_like(activation)
    selections: list[np.ndarray] = []
    for index in range(len(scores)):
        other = np.delete(semantic, index, axis=0)
        competing = other.max(axis=0) if len(other) else np.zeros(768, dtype=np.float32)
        specificity = semantic[index] / (semantic[index] + specificity_weight * competing + 1e-8)
        scores[index] = activation[index] * semantic[index] * specificity
        selected = np.argsort(-scores[index])[:topk_per_class]
        if not np.isfinite(scores[index, selected]).all() or scores[index, selected].max() <= 0:
            raise RuntimeError(f"class {index} has no positive concept-specific circuit score")
        selections.append(selected.astype(np.int64))
    union = np.unique(np.concatenate(selections))
    class_mask = np.zeros((len(selections), len(union)), dtype=np.float32)
    class_direction = np.zeros_like(class_mask)
    lookup = {int(value): index for index, value in enumerate(union)}
    for class_index, selected in enumerate(selections):
        columns = np.asarray([lookup[int(value)] for value in selected])
        class_mask[class_index, columns] = 1.0
        class_direction[class_index, columns] = direction[class_index, selected]
    return union, class_mask, class_direction, scores, selections


def normal_gate_threshold(
    matched: np.ndarray,
    dataset: str,
    model,
    normal_text: torch.Tensor,
    abnormal_text: torch.Tensor,
    quantile: float,
    device: torch.device,
) -> tuple[float, dict]:
    values = []
    margin = TextMargin(model, normal_text, abnormal_text)
    for _, label, sample in tqdm(matched, desc="calibrate normal concept gate", unit="video"):
        if not is_normal_label(dataset, str(label)):
            continue
        with torch.no_grad():
            image = margin.image(torch.from_numpy(sample).to(device))
            score = (image @ abnormal_text.T).max(-1).values - (image @ normal_text.T).max(-1).values
        values.append(score.cpu().numpy())
    all_values = np.concatenate(values)
    threshold = float(np.quantile(all_values, quantile))
    return threshold, {
        "quantile": quantile,
        "threshold": threshold,
        "samples": int(len(all_values)),
        "mean": float(all_values.mean()),
        "std": float(all_values.std()),
    }


def intervention_diagnostics(
    matched: np.ndarray,
    dataset: str,
    class_names: list[str],
    selections: list[np.ndarray],
    directions: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    model,
    normal_text: torch.Tensor,
    abnormal_text: torch.Tensor,
    tail_fraction: float,
    max_videos_per_class: int,
    seed: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    selected_effects: list[list[float]] = [[] for _ in class_names]
    random_effects: list[list[float]] = [[] for _ in class_names]
    off_effects: list[list[float]] = [[] for _ in class_names]
    used = np.zeros(len(class_names), dtype=np.int64)
    all_dims = np.arange(768)
    for _, label, sample in tqdm(matched, desc="counterfactual circuit check", unit="video"):
        if is_normal_label(dataset, str(label)):
            continue
        for class_index in label_classes(dataset, str(label), class_names):
            if used[class_index] >= max_videos_per_class:
                continue
            used[class_index] += 1
            selected = selections[class_index]
            sign = directions[class_index, selected]
            available = np.setdiff1d(all_dims, selected)
            random_dims = rng.choice(available, len(selected), replace=False)
            random_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), len(selected))
            hidden = torch.from_numpy(sample).to(device)
            center_tensor = torch.from_numpy(center).to(device)
            scale_tensor = torch.from_numpy(scale).to(device)

            def views(dims: np.ndarray, signs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
                index = torch.from_numpy(dims).to(device)
                direction = torch.from_numpy(signs).to(device)
                evidence = F.relu(
                    direction * (hidden.index_select(1, index) - center_tensor[index])
                    / scale_tensor[index].clamp_min(1e-6)
                )
                delta = 0.50 * direction * scale_tensor[index] * evidence
                plus = hidden.clone(); minus = hidden.clone()
                plus[:, index] += delta
                minus[:, index] -= 2.0 * delta
                return plus, minus

            with torch.no_grad():
                selected_plus, selected_minus = views(selected, sign.astype(np.float32))
                random_plus, random_minus = views(random_dims, random_sign)
                selected_delta = (
                    F.normalize(TextMargin(model, normal_text, abnormal_text).image(selected_plus), dim=-1)
                    - F.normalize(TextMargin(model, normal_text, abnormal_text).image(selected_minus), dim=-1)
                ) @ abnormal_text.T
                random_delta = (
                    F.normalize(TextMargin(model, normal_text, abnormal_text).image(random_plus), dim=-1)
                    - F.normalize(TextMargin(model, normal_text, abnormal_text).image(random_minus), dim=-1)
                ) @ abnormal_text.T
            count = max(1, int(np.ceil(len(sample) * tail_fraction)))
            selected_target = selected_delta[:, class_index]
            random_target = random_delta[:, class_index]
            selected_effects[class_index].append(float(torch.topk(selected_target, count).values.mean().cpu()))
            random_effects[class_index].append(float(torch.topk(random_target, count).values.mean().cpu()))
            other = torch.cat([selected_delta[:, :class_index], selected_delta[:, class_index + 1:]], dim=-1)
            off_effects[class_index].append(float(other.abs().mean().cpu()))
    per_class = {}
    for index, name in enumerate(class_names):
        selected_mean = float(np.mean(selected_effects[index]))
        random_mean = float(np.mean(random_effects[index]))
        off_mean = float(np.mean(off_effects[index]))
        per_class[name] = {
            "videos": int(used[index]),
            "selected_target_effect": selected_mean,
            "random_target_effect": random_mean,
            "off_target_absolute_effect": off_mean,
            "selected_random_ratio": selected_mean / max(abs(random_mean), 1e-8),
            "target_off_ratio": selected_mean / max(off_mean, 1e-8),
        }
    return {
        "per_class": per_class,
        "selected_target_effect_mean": float(np.mean([np.mean(value) for value in selected_effects])),
        "random_target_effect_mean": float(np.mean([np.mean(value) for value in random_effects])),
        "off_target_absolute_effect_mean": float(np.mean([np.mean(value) for value in off_effects])),
    }


def build_compact_split(
    name: str,
    source_csv: str,
    manifest: str,
    output: Path,
    layer: int,
    union: np.ndarray,
    clean: bool,
    skip_missing: bool,
) -> dict:
    frame = read_feature_csv(source_csv)
    mapping, _ = read_hidden_manifest(manifest)
    feature_dir = output / name / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped, reused = [], [], 0
    cache: dict[str, np.ndarray] = {}
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"build {name} circuit features", unit="crop"):
        clip_path = str(row["path"])
        key = base_key(clip_path)
        if key not in mapping:
            if not skip_missing:
                raise FileNotFoundError(f"missing hidden state for {key}")
            skipped.append([key, str(row["label"]), clip_path, "missing_hidden"])
            continue
        clip = np.load(clip_path, mmap_mode="r")
        if clip.ndim != 2 or clip.shape[1] != 512:
            raise ValueError(f"{clip_path}: expected [T,512], got {clip.shape}")
        target = feature_dir / f"{Path(clip_path).stem}.npy"
        if target.exists() and not clean:
            compact = np.load(target, mmap_mode="r")
            if compact.shape != (len(clip), len(union) + 2):
                raise ValueError(f"stale compact feature {target}: {compact.shape}")
            reused += 1
        else:
            if key not in cache:
                hidden = hidden_layer(mapping, key, layer)
                selected = hidden[:, union]
                mean = hidden.mean(axis=1, keepdims=True)
                variance = hidden.var(axis=1, keepdims=True)
                cache[key] = np.concatenate([selected, mean, variance], axis=1).astype(np.float32)
            compact = resample_feature(cache[key], len(clip))
            np.save(target, compact.astype(np.float32))
        rows.append([clip_path, str(target), str(row["label"]), key, len(clip)])
    csv_path = output / f"{name}.csv"
    write_csv(csv_path, ["clip_path", "neuron_path", "label", "key", "length"], rows)
    write_csv(output / f"{name}_skipped.csv", ["key", "label", "clip_path", "reason"], skipped)
    return {"csv": str(csv_path), "rows": len(rows), "skipped_rows": len(skipped), "reused": reused}


def projection_validation(
    groups: dict[str, pd.DataFrame], mapping: dict[str, str], layer: int,
    margin: TextMargin, videos: int, device: torch.device,
) -> dict:
    cosines = []
    for key, group in tqdm(list(groups.items())[:videos], desc="validate hidden projection", unit="video"):
        if key not in mapping:
            continue
        clip = np.load(str(group.iloc[0]["path"])).astype(np.float32)
        hidden = resample_feature(hidden_layer(mapping, key, layer), len(clip))
        with torch.no_grad():
            projected = F.normalize(margin.image(torch.from_numpy(hidden).to(device)), dim=-1)
            target = F.normalize(torch.from_numpy(clip).to(device), dim=-1)
            cosines.extend((projected * target).sum(-1).cpu().tolist())
    return {
        "mean_cosine": float(np.mean(cosines)),
        "p05_cosine": float(np.quantile(cosines, 0.05)),
        "samples": len(cosines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a baseline-score-free concept neuron circuit atlas.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--train-list", required=True); parser.add_argument("--test-list", required=True)
    parser.add_argument("--train-hidden-manifest", required=True); parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--clip-root", required=True); parser.add_argument("--clip-weight", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=11); parser.add_argument("--topk-per-class", type=int, default=32)
    parser.add_argument("--tail-fraction", type=float, default=0.10); parser.add_argument("--snippets-per-video", type=int, default=256)
    parser.add_argument("--specificity-weight", type=float, default=1.0)
    parser.add_argument("--normal-gate-quantile", type=float, default=0.95)
    parser.add_argument("--diagnostic-videos-per-class", type=int, default=16)
    parser.add_argument("--projection-videos", type=int, default=32)
    parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-missing-hidden", action="store_true"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 0 < args.tail_fraction <= 1 or not 1 <= args.topk_per_class <= 768:
        parser.error("invalid tail fraction or top-k")
    seed_everything(args.seed)
    output = clean_output(args.out_dir, args.clean)
    atlas_path = output / "circuit_atlas.json"
    if atlas_path.exists() and not args.clean:
        complete = all((output / name).exists() for name in ("build_report.json", "train.csv", "test.csv"))
        if complete:
            print(atlas_path.read_text(encoding="utf-8"), flush=True)
            return
        print("incomplete atlas build detected; recomputing metadata and reusing valid compact files", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, tokenize = load_reference_clip(args.clip_root, args.clip_weight, device)
    normal_text = encode_concepts(model, tokenize, NORMAL_CONCEPTS, device)
    class_names = class_names_for(args.dataset)
    abnormal_text = encode_concepts(model, tokenize, class_names, device)
    train_frame = read_feature_csv(args.train_list)
    groups = grouped_rows(train_frame)
    train_mapping, _ = read_hidden_manifest(args.train_hidden_manifest)
    center, scale, positive, negative, matched, statistics = collect_statistics(
        groups, train_mapping, args.dataset, args.layer, args.snippets_per_video,
        args.tail_fraction, class_names,
    )
    gradients = concept_gradients(model, normal_text, abnormal_text, center, scale, device)
    union, class_mask, directions, scores, selections = select_circuits(
        positive, negative, gradients, args.topk_per_class, args.specificity_weight
    )
    threshold, gate_report = normal_gate_threshold(
        matched, args.dataset, model, normal_text, abnormal_text,
        args.normal_gate_quantile, device,
    )
    diagnostics = intervention_diagnostics(
        matched, args.dataset, class_names, selections,
        np.where(positive >= negative, 1.0, -1.0).astype(np.float32),
        center, scale, model, normal_text, abnormal_text, args.tail_fraction,
        args.diagnostic_videos_per_class, args.seed, device,
    )
    projection = projection_validation(
        groups, train_mapping, args.layer,
        TextMargin(model, normal_text, abnormal_text), args.projection_videos, device,
    )
    weights_path = output / "circuit_weights.npz"
    np.savez_compressed(
        weights_path,
        union_indices=union.astype(np.int64), class_mask=class_mask, directions=directions,
        center=center[union], scale=scale[union], ln_weight=model.visual.ln_post.weight.detach().cpu().numpy(),
        ln_bias=model.visual.ln_post.bias.detach().cpu().numpy(), projection=model.visual.proj.detach().cpu().numpy(),
        normal_text=normal_text.detach().cpu().numpy(), abnormal_text=abnormal_text.detach().cpu().numpy(),
        scores=scores[:, union],
    )
    checks = {
        "projection": projection["mean_cosine"] >= 0.90,
        "beats_random": diagnostics["selected_target_effect_mean"] > diagnostics["random_target_effect_mean"],
        "positive_target_effect": diagnostics["selected_target_effect_mean"] > 0.0,
        "concept_specific": diagnostics["selected_target_effect_mean"] > diagnostics["off_target_absolute_effect_mean"],
    }
    atlas = {
        "method": "concept_conditioned_neuron_circuit_routing_v1",
        "dataset": args.dataset, "layer_zero_based": args.layer,
        "class_names": class_names, "normal_concepts": list(NORMAL_CONCEPTS),
        "topk_per_class": args.topk_per_class, "union_width": int(len(union)),
        "selected_indices_per_class": {name: selections[index].tolist() for index, name in enumerate(class_names)},
        "weights_path": weights_path.name, "normal_margin_threshold": threshold,
        "layer_norm_eps": float(getattr(model.visual.ln_post, "eps", 1e-5)),
        "selection_source": "video labels, normal statistics and frozen CLIP text gradients; no baseline anomaly scores",
        "statistics": statistics, "normal_gate": gate_report,
        "projection_validation": projection, "intervention_diagnostics": diagnostics,
        "checks": checks, "gate_passed": bool(all(checks.values())),
    }
    save_json(atlas_path, atlas)
    build_report = {
        "atlas": str(atlas_path), "gate_passed": atlas["gate_passed"],
        "train": build_compact_split(
            "train", args.train_list, args.train_hidden_manifest, output, args.layer,
            union, args.clean, args.skip_missing_hidden,
        ),
        "test": build_compact_split(
            "test", args.test_list, args.test_hidden_manifest, output, args.layer,
            union, args.clean, args.skip_missing_hidden,
        ),
        "input_signature": {
            "train_list_sha256": hashlib.sha256(Path(args.train_list).read_bytes()).hexdigest(),
            "train_manifest_sha256": hashlib.sha256(Path(args.train_hidden_manifest).read_bytes()).hexdigest(),
        },
    }
    save_json(output / "build_report.json", build_report)
    print(json.dumps({"atlas": atlas, "build": build_report}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
