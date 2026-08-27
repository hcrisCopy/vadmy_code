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
  # The sparse expert depends only on the dataset and fixed neuron configuration,
  # so keep its resumable checkpoint outside commit-keyed evaluation directories.
  diverse="$ROOT/diverse_expert_cache/$dataset/active64_seed3407"
  python -m universal_neuron_adapter.train_diverse_expert \
    --train-manifest "$SOURCE/$dataset/data/expert_train.csv" --val-manifest "$SOURCE/$dataset/data/expert_val.csv" \
    --out-dir "$diverse" --active-per-layer 64 --temporal-width 64 --max-epoch 20 \
    --batch-size 8 --lr 0.0003 --weight-decay 0.0001 --sparsity-weight 0.001 \
    --maximum-length 256 --num-workers 4 --seed 3407 --device cuda --resume
  python -m universal_neuron_adapter.export_diverse_expert \
    --manifest "$SOURCE/$dataset/data/test.csv" --expert-model "$diverse/expert_best.pth" \
    --out-dir "$diverse/test" --device cuda
  normality="$ROOT/normality_expert_cache/$dataset/top32_v1"
  python -m universal_neuron_adapter.fit_normality_expert \
    --manifest "$SOURCE/$dataset/data/expert_train.csv" --out-dir "$normality" \
    --active-per-layer 32 --maximum-length 256 --resume
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$SOURCE/$dataset/data/test.csv" --expert-model "$normality/normality_expert.npz" \
    --out-dir "$normality/test"
  for baseline in lagovad desc dsanet; do
    source_base="$SOURCE/$dataset/$baseline"
    target="$OUT/$dataset/$baseline/evaluation"
    python -m universal_neuron_adapter.evaluate \
      --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
      --expert2-manifest "$diverse/test/expert2_scores.csv" \
      --expert3-manifest "$normality/test/expert3_scores.csv" \
      --correction-model "$source_base/correction/model_best.pth" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --correction-weight 0.2 --neuron-weight 0.4 \
      --event-width 51 --event-weight 1.0 \
      --normality-gate-weight 1.5 \
      --normal-suppression-weight 1.5 \
      --persistence-width 15 --persistence-weight 0.75 \
      --gaussian-sigma 1.0 \
      --advance-snippets 1 --device cuda
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

