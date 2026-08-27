#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

RUN_KEY="$(git rev-parse --short HEAD)"
ROOT="../vadmy_data/universal_neuron_adapter"
SOURCE="$ROOT/runs/9d1a066"
OUT="$ROOT/runs/$RUN_KEY"
mkdir -p "$OUT"
printf '%s\n' "$RUN_KEY" > "$ROOT/current_run.txt"
exec > >(tee -a "$OUT/run.log") 2>&1

for dataset in ucf xd; do
  for baseline in lagovad desc dsanet; do
    source_base="$SOURCE/$dataset/$baseline"
    target="$OUT/$dataset/$baseline/evaluation"
    python -m universal_neuron_adapter.evaluate \
      --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --rank-weight 0.5 --event-width 25 --event-weight 0.5 \
      --neuron-weight 0.15
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

