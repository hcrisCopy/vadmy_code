#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
ROOT="../vadmy_data/universal_neuron_adapter/robustness_efficiency/$(git rev-parse --short HEAD)"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

evaluate_setting() {
  local variant="$1" dataset="$2" baseline="$3" width="$4" advance="$5"
  local diverse="../vadmy_data/universal_neuron_adapter/diverse_expert_cache/$dataset/active64_seed3407"
  local normality="../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1"
  local context="../vadmy_data/universal_neuron_adapter/context_student_cache/$dataset/top32_multiscale_seed3407"
  local source_base="$SOURCE/$dataset/$baseline"
  python -m universal_neuron_adapter.evaluate \
    --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
    --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
    --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
    --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
    --expert2-manifest "$diverse/test/expert2_scores.csv" --expert2-train-manifest "$diverse/train/expert2_scores.csv" \
    --expert3-manifest "$normality/test/expert3_scores.csv" --expert3-train-manifest "$normality/train/expert3_scores.csv" \
    --student-manifest "$context/test/student_scores.csv" --student-train-manifest "$context/train/student_scores.csv" \
    --correction-model "$source_base/correction/model_best.pth" --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --baseline "$baseline" --dataset "$dataset" --out-dir "$ROOT/evaluations/$variant/$dataset/$baseline" \
    --frames-per-snippet 16 --event-width "$width" --event-weight 1.0 --normality-smoothing-blend 0.25 \
    --persistence-weight 1.0 --gaussian-sigma 0.5 --advance-snippets "$advance" --device cuda
}

for dataset in ucf xd; do
  for baseline in lagovad desc dsanet; do
    evaluate_setting width_33 "$dataset" "$baseline" 33 1
    evaluate_setting width_41 "$dataset" "$baseline" 41 1
    evaluate_setting width_49 "$dataset" "$baseline" 49 1
    evaluate_setting advance_0 "$dataset" "$baseline" 41 0
    evaluate_setting advance_2 "$dataset" "$baseline" 41 2
  done
done

python -m universal_neuron_adapter.analyze_robustness --root "$ROOT"
printf '%s\n' "$ROOT" > ../vadmy_data/universal_neuron_adapter/current_robustness_efficiency.txt
