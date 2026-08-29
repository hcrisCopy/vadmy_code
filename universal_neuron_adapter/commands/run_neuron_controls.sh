#!/usr/bin/env bash
set -euo pipefail
SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
ROOT="../vadmy_data/universal_neuron_adapter/neuron_controls/$(git rev-parse --short HEAD)"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

for dataset in ucf xd; do
  expert_model="$SOURCE/$dataset/expert/expert_best.pth"
  for control in remove_selected random_matched; do
    seeds=(234)
    if [[ "$control" == "random_matched" ]]; then seeds=(234 3407 2026 17 73); fi
    for seed in "${seeds[@]}"; do
      expert="$ROOT/$control/seed_$seed/$dataset/expert"
      python -m universal_neuron_adapter.export_expert --manifest "$SOURCE/$dataset/data/test.csv" --expert-model "$expert_model" --out-dir "$expert/test" --device cuda --control "$control" --control-seed "$seed"
      context="../vadmy_data/universal_neuron_adapter/context_student_cache/$dataset/top32_multiscale_seed234"
      normality="../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1"
      for baseline in lagovad desc dsanet; do
        source_base="$SOURCE/$dataset/$baseline"
        python -m universal_neuron_adapter.evaluate --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
          --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" --expert-manifest "$expert/test/expert_scores.csv" \
          --expert3-manifest "$normality/test/expert3_scores.csv" --expert3-train-manifest "$normality/train/expert3_scores.csv" \
          --student-manifest "$context/test/student_scores.csv" --student-train-manifest "$context/train/student_scores.csv" \
          --correction-model "$source_base/correction/model_best.pth" --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
          --baseline "$baseline" --dataset "$dataset" --out-dir "$ROOT/$control/seed_$seed/$dataset/$baseline" \
          --frames-per-snippet 16 --event-width 41 --event-weight 1.0 --normality-smoothing-blend 0.25 --persistence-weight 1.0 --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda
      done
    done
  done
done
FULL_ROOT="$(cat ../vadmy_data/universal_neuron_adapter/current_supplementary_run.txt)/ablations/full"
python -m universal_neuron_adapter.analyze_neuron_controls --root "$ROOT" --full-root "$FULL_ROOT" --random-seeds 234 3407 2026 17 73
printf '%s\n' "$ROOT" > ../vadmy_data/universal_neuron_adapter/current_neuron_controls.txt
