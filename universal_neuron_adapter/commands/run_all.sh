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
    guided="$OUT/$dataset/$baseline/guided_correction"
    python -m universal_neuron_adapter.train_guided_correction \
      --baseline-manifest "$source_base/baseline_train/baseline_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
      --train-keys "$SOURCE/$dataset/data/expert_train.csv" \
      --val-keys "$SOURCE/$dataset/data/expert_val.csv" \
      --baseline "$baseline" --dataset "$dataset" --out-dir "$guided" \
      --width 32 --max-epoch 10 --batch-size 32 --lr 3e-4 \
      --weight-decay 1e-4 --maximum-length 256 --num-workers 4 \
      --seed 3407 --device cuda --resume
    python -m universal_neuron_adapter.evaluate \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
      --correction-model "$guided/model_best.pth" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --correction-weight 0.2 --neuron-weight 0.1 \
      --event-width 25 --event-weight 1.0 --device cuda
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

