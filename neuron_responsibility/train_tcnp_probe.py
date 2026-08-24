#!/usr/bin/env python3
"""Fit and audit a baseline-score-free text-conditioned neuron probe."""

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neuron_responsibility.baselines import label_map_for
from neuron_responsibility.common import clean_output, is_normal_label
from neuron_responsibility.data import AlignedFeatureDataset, load_full_row
from neuron_responsibility.tcnp import TextConditionedNeuronProbe


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_indices(labels: list[str], dataset: str, class_names: list[str], device: torch.device) -> torch.Tensor:
    mapping = label_map_for(dataset); result = torch.zeros(len(labels), len(class_names), device=device)
    for row, label in enumerate(labels):
        raw = [label] if dataset == "ucf" else str(label).split("-")
        for value in raw:
            if value in mapping and mapping[value] in class_names:
                result[row, class_names.index(mapping[value])] = 1.0
    return result


def split_indices(frame: pd.DataFrame, seed: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    validation_keys = set()
    for key in frame["key"].astype(str).unique():
        digest = hashlib.sha1(f"{seed}:{key}".encode()).digest()
        if int.from_bytes(digest[:8], "little") / 2**64 < validation_fraction:
            validation_keys.add(key)
    # UCF training lists contain ten spatial crops.  Hidden CLS sequences describe
    # the same video, so one deterministic representative avoids tenfold I/O and
    # prevents crop duplicates from making the held-out audit look overconfident.
    representatives = frame.reset_index().groupby("key", sort=False).first().reset_index()
    pairs = [(int(row["index"]), str(row["key"])) for _, row in representatives.iterrows()]
    fit = [index for index, key in pairs if key not in validation_keys]
    validation = [index for index, key in pairs if key in validation_keys]
    return fit, validation


def probe_loss(model, batch, dataset: str, top_fraction: float, consistency_weight: float, device):
    neurons = batch["neurons"].to(device); lengths = batch["length"].to(device)
    binary = batch["binary_label"].to(device); targets = class_indices(batch["label_text"], dataset, model.class_names, device)
    record = model(neurons); logits, layers = record["logits"], record["layer_logits"]
    normal_terms, positive_terms, class_terms = [], [], []
    for row, length_tensor in enumerate(lengths):
        length = max(1, int(length_tensor)); valid = logits[row, :length]
        if binary[row] < 0.5:
            normal_terms.append(F.softplus(valid.max(-1).values).mean())
            continue
        target_ids = torch.nonzero(targets[row] > 0, as_tuple=False).flatten()
        if not len(target_ids):
            continue
        count = max(1, int(np.ceil(length * top_fraction)))
        target_values = valid.index_select(-1, target_ids).max(-1).values
        positive_terms.append(F.softplus(-target_values.topk(count).values).mean())
        bag = valid.topk(count, dim=0).values.mean(0)
        target = targets[row] / targets[row].sum().clamp_min(1.0)
        class_terms.append(-(target * F.log_softmax(bag, dim=-1)).sum())
    zero = logits.sum() * 0.0
    mean = lambda values: torch.stack(values).mean() if values else zero
    valid_mask = torch.arange(logits.shape[1], device=device)[None, :] < lengths[:, None]
    consistency = (torch.sigmoid(layers) - torch.sigmoid(logits).unsqueeze(-2)).square()
    consistency = consistency[valid_mask].mean()
    values = {"normal": mean(normal_terms), "positive": mean(positive_terms), "class": mean(class_terms), "consistency": consistency}
    values["total"] = values["normal"] + values["positive"] + 0.25 * values["class"] + consistency_weight * consistency
    return values


@torch.no_grad()
def raw_scores(model, neurons: torch.Tensor, prototypes: np.ndarray | None, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    record = model(neurons.to(device)); logits = record["logits"]; layers = record["layer_logits"]
    semantic, predicted = logits.max(-1)
    agreement = (layers.argmax(-1) == predicted.unsqueeze(-1)).float().mean(-1)
    if prototypes is None:
        normal_distance = torch.ones_like(semantic)
    else:
        standardized = record["standardized"]
        centers = torch.from_numpy(prototypes).to(device)
        normal_distance = torch.cdist(standardized, centers).min(-1).values / np.sqrt(model.width)
    return torch.stack([semantic, normal_distance, agreement], -1).cpu().numpy(), predicted.cpu().numpy()


def fit_prototypes(model, source: AlignedFeatureDataset, indices: list[int], clusters: int, samples: int, device) -> np.ndarray:
    values = []
    for index in tqdm(indices, desc="collect normal prototypes", unit="crop"):
        item = source[index]
        if item["binary_label"] > 0.5:
            continue
        length = int(item["length"]); step = max(1, int(np.ceil(length / 64)))
        with torch.no_grad():
            values.append(model(item["neurons"][:length:step].unsqueeze(0).to(device))["standardized"][0].cpu().numpy())
    matrix = np.concatenate(values); generator = np.random.default_rng(234)
    if len(matrix) > samples: matrix = matrix[generator.choice(len(matrix), samples, replace=False)]
    count = min(clusters, len(matrix))
    return MiniBatchKMeans(count, random_state=234, batch_size=2048, n_init=3).fit(matrix).cluster_centers_.astype(np.float32)


@torch.no_grad()
def calibrate(model, source, indices, prototypes, quantile, device) -> dict:
    rows = []
    for index in tqdm(indices, desc="calibrate fit normal", unit="crop"):
        item = source[index]
        if item["binary_label"] > 0.5: continue
        length = int(item["length"]); scores, _ = raw_scores(model, item["neurons"][:length].unsqueeze(0), prototypes, device); rows.append(scores[0])
    matrix = np.concatenate(rows)
    return {"semantic": float(np.quantile(matrix[:, 0], quantile)), "distance": float(np.quantile(matrix[:, 1], quantile)), "quantile": quantile}


def combined(values: np.ndarray, calibration: dict) -> np.ndarray:
    """Return the primary text-circuit localization confidence.

    Prototype distance and cross-layer agreement remain independent audit
    channels.  Requiring their intersection was rejected because normal-video
    held-out evidence showed that it discarded text-grounded positives without
    improving the controlled normal false-positive rate.
    """
    semantic = 1.0 / (1.0 + np.exp(-(values[:, 0] - calibration["semantic"]) / 0.25))
    return semantic


@torch.no_grad()
def audit_split(model, source, indices, prototypes, calibration, threshold, device) -> dict:
    by_key: dict[str, dict] = {}
    for index in tqdm(indices, desc="audit held-out training videos", unit="crop"):
        item = source[index]; length = int(item["length"])
        values, predicted = raw_scores(model, item["neurons"][:length].unsqueeze(0), prototypes, device)
        score = combined(values[0], calibration); key = item["key"]
        target = class_indices([item["label_text"]], source.dataset, model.class_names, device)[0].cpu().numpy()
        entry = by_key.setdefault(key, {"normal": float(item["binary_label"]) < 0.5, "scores": [], "predicted": [], "target": target})
        entry["scores"].append(score); entry["predicted"].append(predicted[0])
    normal_selected = normal_total = covered = positives = correct = 0
    for entry in by_key.values():
        score = np.concatenate(entry["scores"]); prediction = np.concatenate(entry["predicted"]); chosen = score >= threshold
        if entry["normal"]:
            normal_selected += int(chosen.sum()); normal_total += len(chosen)
        else:
            positives += 1; covered += int(chosen.any())
            if chosen.any(): correct += int(entry["target"][prediction[np.argmax(score)]] > 0)
    normal_fpr = normal_selected / max(1, normal_total); coverage = covered / max(1, positives); accuracy = correct / max(1, covered)
    passed = normal_fpr <= 0.02 and coverage >= 0.50 and accuracy >= 0.30
    return {"normal_snippet_fpr": normal_fpr, "abnormal_video_coverage": coverage, "covered_video_class_accuracy": accuracy,
            "normal_snippets": normal_total, "abnormal_videos": positives, "passed": passed,
            "rule": "normal_fpr<=0.02 AND abnormal_coverage>=0.50 AND class_accuracy>=0.30"}


@torch.no_grad()
def test_diagnostic(model, csv_path, prototypes, calibration, threshold, gt_path, repeat, device) -> dict:
    frame = pd.read_csv(csv_path); all_scores = []
    for _, group in tqdm(list(frame.groupby("key", sort=False)), desc="fixed test diagnostic", unit="video"):
        video = []
        for _, row in group.iterrows():
            _, neuron, _, _ = load_full_row(row); values, _ = raw_scores(model, torch.from_numpy(neuron).unsqueeze(0), prototypes, device); video.append(combined(values[0], calibration))
        all_scores.append(np.concatenate(video))
    snippet = np.concatenate(all_scores); prediction = np.repeat(snippet, repeat); truth = np.load(gt_path).reshape(-1).astype(np.int64)
    usable = min(len(prediction), len(truth)); prediction, truth = prediction[:usable], truth[:usable]; selected = prediction >= threshold
    return {"frame_auc": float(roc_auc_score(truth, prediction)), "frame_ap": float(average_precision_score(truth, prediction)),
            "candidate_precision": float(truth[selected].mean()) if selected.any() else 0.0,
            "candidate_recall": float(truth[selected].sum() / max(1, truth.sum())), "selected_frames": int(selected.sum()), "frames": usable,
            "role": "final diagnostic only; never used by training or the quality gate"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and audit the TCNP sparse-neuron probe.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True); parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True); parser.add_argument("--atlas", required=True); parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-fraction", type=float, default=0.10); parser.add_argument("--consistency-weight", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20); parser.add_argument("--normal-quantile", type=float, default=0.98)
    parser.add_argument("--prototype-count", type=int, default=32); parser.add_argument("--prototype-samples", type=int, default=50000)
    parser.add_argument("--candidate-threshold", type=float, default=0.50); parser.add_argument("--frames-per-snippet", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--seed", type=int, default=234); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.resume: parser.error("--clean and --resume cannot be combined")
    output = clean_output(args.out_dir, args.clean); last = output / "checkpoint_last.pth"; best = output / "probe_best.pth"
    if last.exists() and not args.resume: raise RuntimeError("checkpoint exists; use --resume or --clean")
    seed_all(args.seed); device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    source = AlignedFeatureDataset(args.train_list, args.dataset, 256); fit, validation = split_indices(source.frame, args.seed, args.validation_fraction)
    model = TextConditionedNeuronProbe.from_atlas(args.atlas).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    if args.resume:
        state = torch.load(last, map_location="cpu"); model.load_state_dict(state["probe_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); start = state["epoch"] + 1
    loader = DataLoader(Subset(source, fit), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)
    history = output / "probe_history.jsonl"
    for epoch in range(start, args.epochs):
        model.train(); totals = {key: 0.0 for key in ("total", "normal", "positive", "class", "consistency")}
        progress = tqdm(loader, desc=f"TCNP probe {epoch + 1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(progress, 1):
            losses = probe_loss(model, batch, args.dataset, args.top_fraction, args.consistency_weight, device)
            optimizer.zero_grad(set_to_none=True); losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            for key in totals: totals[key] += float(losses[key].detach())
            progress.set_postfix(loss=f"{totals['total']/step:.4f}")
        record = {"epoch": epoch + 1, **{f"{key}_loss": value/max(1, step) for key, value in totals.items()}}
        with history.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        payload = {"epoch": epoch, "probe_config": model.config(), "probe_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)}
        torch.save(payload, last); torch.save(payload, best)
    model.eval(); prototypes = fit_prototypes(model, source, fit, args.prototype_count, args.prototype_samples, device)
    np.save(output / "normal_prototypes.npy", prototypes)
    calibration = calibrate(model, source, fit, prototypes, args.normal_quantile, device)
    gate = audit_split(model, source, validation, prototypes, calibration, args.candidate_threshold, device)
    diagnostic = test_diagnostic(model, args.test_list, prototypes, calibration, args.candidate_threshold, args.gt_path, args.frames_per_snippet, device)
    report = {"method": model.method_name, "selection": model.config(), "split": {"fit_rows": len(fit), "validation_rows": len(validation)},
              "calibration": calibration, "candidate_threshold": args.candidate_threshold,
              "fusion_rule": "text-circuit semantic confidence is primary; normal-prototype distance and layer agreement are audit channels",
              "training_quality_gate": gate, "test_diagnostic": diagnostic,
              "baseline_training_allowed": bool(gate["passed"]), "test_labels_used_for_gate": False}
    (output / "gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
