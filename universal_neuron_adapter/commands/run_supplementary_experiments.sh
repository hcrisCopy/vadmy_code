#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

SOURCE="../vadmy_data/universal_neuron_adapter/runs/9d1a066"
ROOT="../vadmy_data/universal_neuron_adapter/supplementary/$(git rev-parse --short HEAD)"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

for dataset in ucf xd; do
  work="../vad_data/work_${dataset}"
  train_hidden="$work/clip_hidden_stride16_train_8gpu/manifest.csv"
  test_hidden="$work/clip_hidden_stride16_test_8gpu/manifest.csv"
  missing=()
  if [[ "$dataset" == "ucf" ]]; then
    train_csv="$work/ucf_train_local.csv"; test_csv="$work/ucf_test_local.csv"
  else
    train_csv="$work/xd_train_local.csv"; test_csv="$work/xd_test_local.csv"; missing=(--skip-missing-hidden)
  fi
  python -m universal_neuron_adapter.data --dataset "$dataset" --train-csv "$train_csv" --test-csv "$test_csv" \
    --train-hidden-manifest "$train_hidden" --test-hidden-manifest "$test_hidden" \
    --out-dir "$ROOT/audit/$dataset" --seed 234 --val-fraction 0.2 "${missing[@]}"
done

evaluate_variant() {
  local variant="$1" dataset="$2" baseline="$3"; shift 3
  local context="../vadmy_data/universal_neuron_adapter/context_student_cache/$dataset/top32_multiscale_seed234"
  local normality="../vadmy_data/universal_neuron_adapter/normality_expert_cache/$dataset/top32_signed_v1"
  local source_base="$SOURCE/$dataset/$baseline"
  python -m universal_neuron_adapter.evaluate \
    --baseline-train-manifest "$source_base/baseline_train/baseline_scores.csv" \
    --baseline-manifest "$source_base/baseline_test/baseline_scores.csv" \
    --expert-train-manifest "$SOURCE/$dataset/expert/train/expert_scores.csv" \
    --expert-manifest "$SOURCE/$dataset/expert/test/expert_scores.csv" \
    --expert3-manifest "$normality/test/expert3_scores.csv" --expert3-train-manifest "$normality/train/expert3_scores.csv" \
    --student-manifest "$context/test/student_scores.csv" --student-train-manifest "$context/train/student_scores.csv" \
    --correction-model "$source_base/correction/model_best.pth" --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --baseline "$baseline" --dataset "$dataset" --out-dir "$ROOT/ablations/$variant/$dataset/$baseline" \
    --frames-per-snippet 16 --event-width 41 --event-weight 1.0 --normality-smoothing-blend 0.25 \
    --persistence-weight 1.0 --gaussian-sigma 0.5 --advance-snippets 0.5 --device cuda "$@"
}

for dataset in ucf xd; do
  for baseline in lagovad desc dsanet vadclip; do
    evaluate_variant baseline "$dataset" "$baseline" --disable-correction --disable-agreement --disable-event-gate --disable-video-suppression --disable-temporal
    evaluate_variant correction "$dataset" "$baseline" --disable-agreement --disable-event-gate --disable-video-suppression --disable-temporal
    evaluate_variant agreement "$dataset" "$baseline" --disable-event-gate --disable-video-suppression --disable-temporal
    evaluate_variant event_gate "$dataset" "$baseline" --disable-video-suppression --disable-temporal
    evaluate_variant video_suppression "$dataset" "$baseline" --disable-temporal
    evaluate_variant full "$dataset" "$baseline"
  done
done

python -m universal_neuron_adapter.analyze_results \
  --results-root "$ROOT/ablations/full" --ucf-gt ../vadmy_data/annotations/ucf/gt.npy \
  --xd-gt ../vadmy_data/annotations/xd/gt.npy --out-dir "$ROOT/statistics" --bootstrap-repeats 200 --seed 234
python -m universal_neuron_adapter.visualize_experiments \
  --experiment-root "$ROOT" --ucf-gt ../vadmy_data/annotations/ucf/gt.npy \
  --xd-gt ../vadmy_data/annotations/xd/gt.npy --out-dir "$ROOT/figures"

printf '%s\n' "$ROOT" > ../vadmy_data/universal_neuron_adapter/current_supplementary_run.txt
