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
  python -m universal_neuron_adapter.consensus_evaluate \
    --desc-train "$SOURCE/$dataset/desc/baseline_train/baseline_scores.csv" \
    --dsanet-train "$SOURCE/$dataset/dsanet/baseline_train/baseline_scores.csv" \
    --desc-test "$SOURCE/$dataset/desc/baseline_test/baseline_scores.csv" \
    --dsanet-test "$SOURCE/$dataset/dsanet/baseline_test/baseline_scores.csv" \
    --lagovad-test "$SOURCE/$dataset/lagovad/baseline_test/baseline_scores.csv" \
    --expert-test "$SOURCE/$dataset/expert/test/expert_scores.csv" \
    --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --dataset "$dataset" --out-root "$OUT/$dataset" \
    --rank-weight 0.5 --event-width 25 --event-weight 0.5 \
    --neuron-weight 0.15 --frames-per-snippet 16
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

