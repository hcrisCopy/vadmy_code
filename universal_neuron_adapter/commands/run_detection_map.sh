#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cat ../vadmy_data/universal_neuron_adapter/current_supplementary_run.txt)"
SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"

for dataset in ucf xd; do
  if [[ "$dataset" == "ucf" ]]; then
    weight="../vadmy_data/model/DSANet/model_ucf.pth"
    segments="baseline/DSANet/list/gt_segment_ucf.npy"
    labels="baseline/DSANet/list/gt_label_ucf.npy"
  else
    weight="../vadmy_data/model/DSANet/model_xd.pth"
    segments="baseline/DSANet/list/gt_segment.npy"
    labels="baseline/DSANet/list/gt_label.npy"
  fi
  cache="$ROOT/detection_map/$dataset/baseline_cache"
  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet --baseline-root baseline/DSANet --baseline-weight "$weight" \
    --dataset "$dataset" --manifest "$SOURCE/$dataset/data/test.csv" --split test \
    --out-dir "$cache" --device cuda
  python -m universal_neuron_adapter.evaluate_detection_map \
    --dataset "$dataset" --baseline-manifest "$cache/baseline_scores.csv" \
    --evaluation-manifest "$ROOT/ablations/full/$dataset/dsanet/per_video.csv" \
    --segment-gt "$segments" --label-gt "$labels" --dsanet-root baseline/DSANet \
    --out-path "$ROOT/detection_map/$dataset/metrics.json"
done
