#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
ROOT="${SEED_STUDY_ROOT:-../vadmy_data/universal_neuron_adapter/seed_study/$(git rev-parse --short HEAD)}"
SEEDS=(234 3407 2026)
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

for seed in "${SEEDS[@]}"; do
  for dataset in ucf xd; do
    data="$SOURCE/$dataset/data"
    primary="$ROOT/seed_$seed/$dataset/primary"
    normality="../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1"
    context="$ROOT/seed_$seed/$dataset/context"
    python -m universal_neuron_adapter.train_expert --train-manifest "$data/expert_train.csv" --val-manifest "$data/expert_val.csv" \
      --out-dir "$primary" --active-per-layer 32 --temporal-width 64 --max-epoch 20 --batch-size 8 \
      --lr 0.0003 --weight-decay 0.0001 --sparsity-weight 0.001 --maximum-length 256 --num-workers 4 --seed "$seed" --device cuda --resume
    # The correction head selects checkpoints on the held-out training-only
    # validation split, so its score cache must cover both disjoint key sets.
    python -m universal_neuron_adapter.export_expert --manifest "$data/train_all.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/train" --device cuda
    python -m universal_neuron_adapter.export_expert --manifest "$data/test.csv" --expert-model "$primary/expert_best.pth" --out-dir "$primary/test" --device cuda
    python -m universal_neuron_adapter.fit_context_student --manifest "$data/expert_train.csv" \
      --expert-manifest "$primary/train/expert_scores.csv" --normality-manifest "$normality/train/expert3_scores.csv" \
      --normality-model "$normality/normality_expert.npz" --out-dir "$context" --normal-samples 32 \
      --positive-fraction 0.05 --epochs 20 --seed "$seed" --resume
    python -m universal_neuron_adapter.export_context_student --manifest "$data/expert_train.csv" --student-model "$context/context_student.npz" --normality-model "$normality/normality_expert.npz" --out-dir "$context/train"
    python -m universal_neuron_adapter.export_context_student --manifest "$data/test.csv" --student-model "$context/context_student.npz" --normality-model "$normality/normality_expert.npz" --out-dir "$context/test"
    for baseline in lagovad desc dsanet vadclip; do
      source_base="$SOURCE/$dataset/$baseline"; correction="$ROOT/seed_$seed/$dataset/$baseline/correction"
      python -m universal_neuron_adapter.train_correction --baseline-manifest "$source_base/baseline_train/baseline_scores.csv" \
        --expert-manifest "$primary/train/expert_scores.csv" --train-keys "$data/expert_train.csv" --val-keys "$data/expert_val.csv" \
        --baseline "$baseline" --dataset "$dataset" --out-dir "$correction" --width 32 --max-epoch 15 --batch-size 32 \
        --lr 0.0003 --weight-decay 0.0001 --maximum-length 256 --num-workers 4 --seed "$seed" --device cuda --resume
      python -m universal_neuron_adapter.evaluate --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
        --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" --expert-train-manifest "$primary/train/expert_scores.csv" \
        --expert-manifest "$primary/test/expert_scores.csv" --expert3-manifest "$normality/test/expert3_scores.csv" \
        --expert3-train-manifest "$normality/train/expert3_scores.csv" --student-manifest "$context/test/student_scores.csv" \
        --student-train-manifest "$context/train/student_scores.csv" --correction-model "$correction/model_best.pth" \
        --gt-path "../vadmy_data/annotations/$dataset/gt.npy" --baseline "$baseline" --dataset "$dataset" \
        --out-dir "$ROOT/seed_$seed/$dataset/$baseline/evaluation" --frames-per-snippet 16 --event-width 41 \
        --event-weight 1.0 --normality-smoothing-blend 0.25 --persistence-weight 1.0 --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda
    done
  done
done
python -m universal_neuron_adapter.analyze_seed_study --root "$ROOT" --seeds "${SEEDS[@]}"
printf '%s\n' "$ROOT" > ../vadmy_data/universal_neuron_adapter/current_seed_study.txt
