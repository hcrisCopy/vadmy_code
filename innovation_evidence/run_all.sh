#!/usr/bin/env bash
set -euo pipefail

ROOT="../vadmy_data/universal_neuron_adapter"
SOURCE="$ROOT/runs/9d1a066"
NORMALITY="$ROOT/normality_expert_cache"
CONTEXT="$ROOT/context_student_cache"
PROBES="$ROOT/interpretability_ablations/3f04e82/probes"
OUTPUT="../vadmy_data/innovation_evidence/$(git rev-parse --short HEAD)"
ANNOTATIONS="../vadmy_data/annotations"

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/run.log") 2>&1

python -m innovation_evidence.innovation1_neurons \
  --probe-root "$PROBES" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --out-dir "$OUTPUT/innovation1" --clean
python -m innovation_evidence.innovation2_spectral \
  --source-root "$SOURCE" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --annotation-root "$ANNOTATIONS" --out-dir "$OUTPUT/innovation2" \
  --frames-per-snippet 16 --clean

evaluate_consensus() {
  local method="$1" dataset="$2" baseline="$3"
  local module="universal_neuron_adapter.evaluate"
  if [[ "$method" == "uniform" ]]; then
    module="innovation_evidence.evaluate_uniform_consensus"
  fi
  local source_base="$SOURCE/$dataset/$baseline"
  local normality="$NORMALITY/$dataset/top32_signed_v1"
  local context="$CONTEXT/$dataset/top32_multiscale_seed234"
  python -m "$module" \
    --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
    --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
    --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
    --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
    --expert3-manifest "$normality/test/expert3_scores.csv" \
    --expert3-train-manifest "$normality/train/expert3_scores.csv" \
    --student-manifest "$context/test/student_scores.csv" \
    --student-train-manifest "$context/train/student_scores.csv" \
    --correction-model "$source_base/correction/model_best.pth" \
    --gt-path "$ANNOTATIONS/$dataset/gt.npy" \
    --baseline "$baseline" --dataset "$dataset" \
    --out-dir "$OUTPUT/innovation2_evaluations/$method/$dataset/$baseline" \
    --frames-per-snippet 16 --event-width 41 --event-weight 1.0 \
    --normality-smoothing-blend 0.25 --persistence-weight 1.0 \
    --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda
}

for dataset in ucf xd; do
  for baseline in lagovad desc dsanet; do
    evaluate_consensus uniform "$dataset" "$baseline"
    evaluate_consensus spectral "$dataset" "$baseline"
  done
done
python -m innovation_evidence.compare_spectral_full \
  --evaluation-root "$OUTPUT/innovation2_evaluations" \
  --out-dir "$OUTPUT/innovation2"

python -m innovation_evidence.innovation3_asymmetry \
  --source-root "$SOURCE" --normality-root "$NORMALITY" --context-root "$CONTEXT" \
  --annotation-root "$ANNOTATIONS" --out-dir "$OUTPUT/innovation3" \
  --frames-per-snippet 16 --clean

printf '%s\n' "$OUTPUT" > ../vadmy_data/innovation_evidence/current_run.txt
