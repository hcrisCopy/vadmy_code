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
  normality="$ROOT/normality_expert_cache/$dataset/top32_signed_v1"
  python -m universal_neuron_adapter.fit_normality_expert \
    --manifest "$SOURCE/$dataset/data/expert_train.csv" --out-dir "$normality" \
    --active-per-layer 32 --maximum-length 256 --resume
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$SOURCE/$dataset/data/test.csv" --expert-model "$normality/normality_expert.npz" \
    --out-dir "$normality/test"
  python -m universal_neuron_adapter.export_normality_expert \
    --manifest "$SOURCE/$dataset/data/expert_train.csv" --expert-model "$normality/normality_expert.npz" \
    --out-dir "$normality/train"
  context="$ROOT/context_student_cache/$dataset/top32_multiscale_seed234"
  python -m universal_neuron_adapter.fit_context_student \
    --manifest "$SOURCE/$dataset/data/expert_train.csv" \
    --expert-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
    --normality-manifest "$normality/train/expert3_scores.csv" \
    --normality-model "$normality/normality_expert.npz" \
    --out-dir "$context" --normal-samples 32 --positive-fraction 0.05 \
    --epochs 20 --seed 234 --resume
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$SOURCE/$dataset/data/test.csv" \
    --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/test"
  python -m universal_neuron_adapter.export_context_student \
    --manifest "$SOURCE/$dataset/data/expert_train.csv" \
    --student-model "$context/context_student.npz" \
    --normality-model "$normality/normality_expert.npz" --out-dir "$context/train"
  diverse="$context"
  for baseline in lagovad desc dsanet; do
    source_base="$SOURCE/$dataset/$baseline"
    target="$OUT/$dataset/$baseline/evaluation"
    python -m universal_neuron_adapter.evaluate \
      --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
      --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
      --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
      --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
      --expert2-manifest "$diverse/test/student_scores.csv" \
      --expert2-train-manifest "$diverse/train/student_scores.csv" \
      --expert3-manifest "$normality/test/expert3_scores.csv" \
      --expert3-train-manifest "$normality/train/expert3_scores.csv" \
      --student-manifest "$context/test/student_scores.csv" \
      --student-train-manifest "$context/train/student_scores.csv" \
      --correction-model "$source_base/correction/model_best.pth" \
      --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
      --baseline "$baseline" --dataset "$dataset" \
      --out-dir "$target" --frames-per-snippet 16 \
      --event-width 41 --event-weight 1.0 \
      --normality-smoothing-blend 0.25 \
      --persistence-weight 1.0 \
      --gaussian-sigma 0.5 \
      --advance-snippets 0.5 --device cuda
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"
