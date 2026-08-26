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
  expert_out="$OUT/$dataset/consensus_expert"
  python -m universal_neuron_adapter.train_consensus_expert \
    --train-keys "$SOURCE/$dataset/data/expert_train.csv" \
    --val-keys "$SOURCE/$dataset/data/expert_val.csv" \
    --desc-manifest "$SOURCE/$dataset/desc/baseline_train/baseline_scores.csv" \
    --dsanet-manifest "$SOURCE/$dataset/dsanet/baseline_train/baseline_scores.csv" \
    --dataset "$dataset" --out-dir "$expert_out" \
    --max-epoch 15 --batch-size 8 --lr 3e-4 --maximum-length 256 \
    --num-workers 4 --seed 234 --device cuda
  python -m universal_neuron_adapter.export_consensus_expert \
    --manifest "$SOURCE/$dataset/data/test.csv" \
    --model "$expert_out/model_best.pth" \
    --out-dir "$expert_out/test" --device cuda
done

for dataset in ucf xd; do
  for baseline in lagovad desc dsanet; do
    source_base="$SOURCE/$dataset/$baseline"
    target="$OUT/$dataset/$baseline/evaluation"
    python -m universal_neuron_adapter.evaluate \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-manifest "$OUT/$dataset/consensus_expert/test/expert_scores.csv" \
      --correction-model "$source_base/correction/model_best.pth" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --correction-weight 0.2 --neuron-weight 0.1 --device cuda
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"
