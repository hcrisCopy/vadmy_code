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
  selected_root="$ROOT/selected_cache/$dataset"
  python -m universal_neuron_adapter.cache_selected_neurons --manifest "$SOURCE/$dataset/data/train_all.csv" --selection "$SOURCE/$dataset/expert/selected_neurons.json" --out-dir "$selected_root/train"
  python -m universal_neuron_adapter.cache_selected_neurons --manifest "$SOURCE/$dataset/data/test.csv" --selection "$SOURCE/$dataset/expert/selected_neurons.json" --out-dir "$selected_root/test"
  semantic="$OUT/$dataset/semantic_probe"
  python -m universal_neuron_adapter.train_semantic_probe \
    --selected-manifest "$selected_root/train/selected_manifest.csv" \
    --train-keys "$SOURCE/$dataset/data/expert_train.csv" --val-keys "$SOURCE/$dataset/data/expert_val.csv" \
    --dataset "$dataset" --out-dir "$semantic" --max-epoch 20 --batch-size 32 \
    --lr 0.0003 --maximum-length 256 --num-workers 4 --seed 3407 --device cuda --resume
  for baseline in lagovad desc dsanet; do
    source_base="$SOURCE/$dataset/$baseline"
    target="$OUT/$dataset/$baseline/evaluation"
    python -m universal_neuron_adapter.evaluate \
      --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
      --correction-model "$source_base/correction/model_best.pth" \
      --semantic-model "$semantic/model_best.pth" \
      --selected-manifest "$selected_root/test/selected_manifest.csv" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --correction-weight 0.2 --neuron-weight 0.1 \
      --event-width 51 --event-weight 1.0 \
      --normal-suppression-weight 1.5 \
      --persistence-width 15 --persistence-weight 0.75 \
      --semantic-weight 0.1 --device cuda
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

