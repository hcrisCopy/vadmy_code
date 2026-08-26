#!/usr/bin/env bash
set -euo pipefail

BASELINE="$1"
DATASET="$2"

if [[ "$DATASET" == "ucf" ]]; then
  TRAIN_CSV="../vad_data/work_ucf/ucf_train_local.csv"
  TEST_CSV="../vad_data/work_ucf/ucf_test_local.csv"
  TRAIN_HIDDEN="../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv"
  TEST_HIDDEN="../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv"
  GT="../vadmy_data/annotations/ucf/gt.npy"
  GT_SEGMENT="../vadmy_data/annotations/ucf/gt_segment.npy"
  GT_LABEL="../vadmy_data/annotations/ucf/gt_label.npy"
  BATCH_SIZE=64
  MAX_EPOCH=10
  MISSING_OPTION=()
else
  TRAIN_CSV="../vad_data/work_xd/xd_train_local.csv"
  TEST_CSV="../vad_data/work_xd/xd_test_local.csv"
  TRAIN_HIDDEN="../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv"
  TEST_HIDDEN="../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv"
  GT="../vadmy_data/annotations/xd/gt.npy"
  GT_SEGMENT="../vadmy_data/annotations/xd/gt_segment.npy"
  GT_LABEL="../vadmy_data/annotations/xd/gt_label.npy"
  BATCH_SIZE=96
  MAX_EPOCH=10
  MISSING_OPTION=(--skip-missing-hidden)
fi

case "$BASELINE" in
  dsanet)
    BASELINE_ROOT="baseline/DSANet"
    if [[ "$DATASET" == "ucf" ]]; then WEIGHT="../vadmy_data/model/DSANet/model_ucf.pth"; LR="7e-5"; else WEIGHT="../vadmy_data/model/DSANet/model_xd.pth"; LR="1e-5"; fi
    WEIGHT_ARGS=(--baseline-weight "$WEIGHT")
    WEIGHT_DECAY="0"
    ;;
  desc)
    BASELINE_ROOT="baseline/DeSC"
    if [[ "$DATASET" == "ucf" ]]; then
      SENSITIVITY="../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth"
      CONSISTENCY="../vadmy_data/model/DeSC/ucf_consistency_stream.pth"
      LR="5e-5"
    else
      SENSITIVITY="../vadmy_data/model/DeSC/xd_sensitivity_stream.pth"
      CONSISTENCY="../vadmy_data/model/DeSC/xd_consistency_stream.pth"
      LR="1e-5"
    fi
    WEIGHT_ARGS=(--sensitivity-weight "$SENSITIVITY" --consistency-weight "$CONSISTENCY")
    WEIGHT_DECAY="1e-5"
    ;;
  lagovad)
    BASELINE_ROOT="baseline/LaGoVAD-PreVAD"
    WEIGHT_ARGS=(--baseline-weight "../vadmy_data/model/LaGoVAD/best.ckpt")
    LR="5e-5"
    WEIGHT_DECAY="0"
    BATCH_SIZE=128
    MAX_EPOCH=40
    ;;
  *)
    echo "unsupported baseline: $BASELINE" >&2
    exit 2
    ;;
esac

# All three Shift experiments share exactly these preparation artifacts.
SHARED="../vadmy_data/shift_residual_head_tuning/${DATASET}/${BASELINE}"
OUT="../vadmy_data/shift_u_dual_head_tuning/${DATASET}/${BASELINE}"

python -m shift_residual_head_tuning.score_baseline \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --source-train-csv "$TRAIN_CSV" \
  --out-dir "$SHARED/pseudo_scores" \
  --device cuda

python -m shift_residual_head_tuning.select_shift_neurons \
  --dataset "$DATASET" \
  --source-train-csv "$TRAIN_CSV" \
  --hidden-manifest "$TRAIN_HIDDEN" \
  --pseudo-csv "$SHARED/pseudo_scores/group_scores.csv" \
  --out-dir "$SHARED/selection" \
  --top-p 0.10 \
  --topk-per-layer 64 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv "$TRAIN_CSV" \
  --hidden-manifest "$TRAIN_HIDDEN" \
  --neuron-json "$SHARED/selection/selected_neurons.json" \
  --out-dir "$SHARED/aligned/train" \
  --out-csv "$SHARED/aligned_train.csv" \
  "${MISSING_OPTION[@]}"

python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv "$TEST_CSV" \
  --hidden-manifest "$TEST_HIDDEN" \
  --neuron-json "$SHARED/selection/selected_neurons.json" \
  --out-dir "$SHARED/aligned/test" \
  --out-csv "$SHARED/aligned_test.csv"

RESUME_OPTION=()
if [[ -f "$OUT/training/checkpoint_last.pth" ]]; then RESUME_OPTION=(--resume); fi
python -m shift_u_dual_head_tuning.train \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --train-list "$SHARED/aligned_train.csv" \
  --val-list "$SHARED/aligned_test.csv" \
  --gt-path "$GT" \
  --neuron-json "$SHARED/selection/selected_neurons.json" \
  --out-dir "$OUT/training" \
  --max-epoch "$MAX_EPOCH" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --hidden-width 1024 \
  --trunk-depth 2 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed 234 \
  --device cuda \
  "${RESUME_OPTION[@]}"

python -m shift_u_dual_head_tuning.evaluate \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --test-list "$SHARED/aligned_test.csv" \
  --model-path "$OUT/training/model_best.pth" \
  --gt-path "$GT" \
  --gt-segment-path "$GT_SEGMENT" \
  --gt-label-path "$GT_LABEL" \
  --out-dir "$OUT/evaluation" \
  --frames-per-snippet 16 \
  --temperature 0 \
  --device cuda

python -m shift_u_dual_head_tuning.visualize_diagnostics \
  --selection-dir "$SHARED/selection" \
  --training-dir "$OUT/training" \
  --out-dir "$OUT/diagnostics"
