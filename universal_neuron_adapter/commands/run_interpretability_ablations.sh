#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
ROOT="../vadmy_data/universal_neuron_adapter/interpretability_ablations/$(git rev-parse --short HEAD)"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

for dataset in ucf xd; do
  data="$SOURCE/$dataset/data"
  primary_model="$SOURCE/$dataset/expert/expert_best.pth"
  probe_root="$ROOT/probes/$dataset"
  python -m universal_neuron_adapter.neuron_probe_ablation \
    --train-manifest "$data/expert_train.csv" \
    --val-manifest "$data/expert_val.csv" \
    --test-manifest "$data/test.csv" \
    --expert-model "$primary_model" --dataset "$dataset" \
    --out-dir "$probe_root" --seed 234 \
    --random-seeds 234 3407 2026 17 73 --resume

  discovery_root="$ROOT/discovery_size/$dataset"
  python -m universal_neuron_adapter.make_discovery_subsets \
    --manifest "$data/expert_train.csv" --out-dir "$discovery_root/subsets" \
    --fractions 0.25 0.5 1.0 --seed 234 --resume

  for fraction in 025 050; do
    model_root="$discovery_root/fraction_$fraction"
    python -m universal_neuron_adapter.train_primary_expert \
      --train-manifest "$discovery_root/subsets/fraction_$fraction.csv" \
      --val-manifest "$data/expert_val.csv" --out-dir "$model_root" \
      --active-per-layer 32 --temporal-width 64 --max-epoch 20 \
      --batch-size 8 --lr 0.0003 --weight-decay 0.0001 \
      --sparsity-weight 0.001 --maximum-length 256 --num-workers 4 \
      --seed 234 --device cuda --resume
    python -m universal_neuron_adapter.evaluate_primary_expert \
      --manifest "$data/test.csv" --expert-model "$model_root/expert_best.pth" \
      --out-file "$model_root/test_metrics.json" --batch-size 8 \
      --maximum-length 256 --num-workers 4 --device cuda --resume
  done

  full_root="$discovery_root/fraction_100"
  mkdir -p "$full_root"
  cp "$SOURCE/$dataset/expert/selected_neurons.json" "$full_root/selected_neurons.json"
  python -m universal_neuron_adapter.evaluate_primary_expert \
    --manifest "$data/test.csv" --expert-model "$primary_model" \
    --out-file "$full_root/test_metrics.json" --batch-size 8 \
    --maximum-length 256 --num-workers 4 --device cuda --resume
  python -m universal_neuron_adapter.analyze_discovery_size \
    --root "$discovery_root" \
    --full-selected "$SOURCE/$dataset/expert/selected_neurons.json" \
    --out-dir "$discovery_root/analysis" --dataset "$dataset"
done

printf '%s\n' "$ROOT" > ../vadmy_data/universal_neuron_adapter/current_interpretability_ablations.txt
echo "[done] interpretability ablations: $ROOT"
