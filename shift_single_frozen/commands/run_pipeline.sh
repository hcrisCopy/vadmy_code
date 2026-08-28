#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

BASELINE="$1"
DATASET="$2"
SEED=234

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
    if [[ "$DATASET" == "ucf" ]]; then
      WEIGHT="../vadmy_data/model/DSANet/model_ucf.pth"
      LR="7e-5"
    else
      WEIGHT="../vadmy_data/model/DSANet/model_xd.pth"
      LR="1e-5"
    fi
    WEIGHT_ARGS=(--baseline-weight "$WEIGHT")
    WEIGHT_DECAY="0"
    ;;
  desc)
    BASELINE_ROOT="baseline/DeSC"
    if [[ "$DATASET" == "ucf" ]]; then
      SENSITIVITY="../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth"
      CONSISTENCY="../vadmy_data/model/DeSC/ucf_consistency_stream.pth"
      WEIGHT_DECAY="1e-5"
    else
      SENSITIVITY="../vadmy_data/model/DeSC/xd_sensitivity_stream.pth"
      CONSISTENCY="../vadmy_data/model/DeSC/xd_consistency_stream.pth"
      WEIGHT_DECAY="1e-3"
    fi
    # DeSC paper: the Temporal Sensitivity stream is optimized at 1e-3.
    LR="1e-3"
    WEIGHT_ARGS=(--sensitivity-weight "$SENSITIVITY" --consistency-weight "$CONSISTENCY")
    ;;
  lagovad)
    BASELINE_ROOT="baseline/LaGoVAD-PreVAD"
    WEIGHT_ARGS=(--baseline-weight "../vadmy_data/model/LaGoVAD/best.ckpt")
    LR="5e-5"
    WEIGHT_DECAY="0"
    BATCH_SIZE=128
    MAX_EPOCH=40
    # Match the seed recorded in the released LaGoVAD config.yaml.
    SEED=2024
    ;;
  *)
    echo "unsupported baseline: $BASELINE" >&2
    exit 2
    ;;
esac

# Scores and neuron selections are intentionally isolated by both dataset and baseline.
RUN_NAME="$BASELINE"
if [[ "$BASELINE" == "desc" ]]; then
  # Keep corrected decoupled-stream products separate from the previous
  # two-stream-injection implementation; never overwrite experimental data.
  RUN_NAME="desc_sensitivity_v2"
fi
OUT="../vadmy_data/shift_single_frozen/${DATASET}/${RUN_NAME}"

python -m shift_single_frozen.provenance prepare-score \
  --baseline "$BASELINE" \
  --dataset "$DATASET" \
  --source-train-csv "$TRAIN_CSV" \
  "${WEIGHT_ARGS[@]}" \
  --out-dir "$OUT/pseudo_scores"

python -m shift_residual_head_tuning.score_baseline \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --source-train-csv "$TRAIN_CSV" \
  --out-dir "$OUT/pseudo_scores" \
  --device cuda

python -m shift_residual_head_tuning.select_shift_neurons \
  --dataset "$DATASET" \
  --source-train-csv "$TRAIN_CSV" \
  --hidden-manifest "$TRAIN_HIDDEN" \
  --pseudo-csv "$OUT/pseudo_scores/group_scores.csv" \
  --out-dir "$OUT/selection" \
  --top-p 0.10 \
  --topk-per-layer 64 \
  --normal-stat-snippets-per-video 256 \
  --sigma-min 1e-6

python -m shift_single_frozen.provenance seal-selection \
  --baseline "$BASELINE" \
  --dataset "$DATASET" \
  --score-provenance "$OUT/pseudo_scores/score_provenance.json" \
  --neuron-json "$OUT/selection/selected_neurons.json" \
  --out-path "$OUT/selection/selection_provenance.json"

python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv "$TRAIN_CSV" \
  --hidden-manifest "$TRAIN_HIDDEN" \
  --neuron-json "$OUT/selection/selected_neurons.json" \
  --out-dir "$OUT/aligned/train" \
  --out-csv "$OUT/aligned_train.csv" \
  "${MISSING_OPTION[@]}"

python -m shift_residual_head_tuning.build_aligned_features \
  --source-csv "$TEST_CSV" \
  --hidden-manifest "$TEST_HIDDEN" \
  --neuron-json "$OUT/selection/selected_neurons.json" \
  --out-dir "$OUT/aligned/test" \
  --out-csv "$OUT/aligned_test.csv"

RESUME_OPTION=()
if [[ -f "$OUT/training/checkpoint_last.pth" ]]; then RESUME_OPTION=(--resume); fi
python -m shift_single_frozen.train \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --train-list "$OUT/aligned_train.csv" \
  --val-list "$OUT/aligned_test.csv" \
  --gt-path "$GT" \
  --neuron-json "$OUT/selection/selected_neurons.json" \
  --selection-provenance "$OUT/selection/selection_provenance.json" \
  --out-dir "$OUT/training" \
  --max-epoch "$MAX_EPOCH" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --residual-hidden-width 1024 \
  --residual-depth 3 \
  --frames-per-snippet 16 \
  --dsanet-ucf-eval-samples 1280 \
  --num-workers 4 \
  --seed "$SEED" \
  --device cuda \
  "${RESUME_OPTION[@]}"

python -m shift_single_frozen.evaluate \
  --baseline "$BASELINE" \
  --baseline-root "$BASELINE_ROOT" \
  "${WEIGHT_ARGS[@]}" \
  --dataset "$DATASET" \
  --test-list "$OUT/aligned_test.csv" \
  --model-path "$OUT/training/model_best.pth" \
  --gt-path "$GT" \
  --gt-segment-path "$GT_SEGMENT" \
  --gt-label-path "$GT_LABEL" \
  --out-dir "$OUT/evaluation" \
  --frames-per-snippet 16 \
  --temperature 0 \
  --device cuda

python -m shift_single_frozen.visualize_diagnostics \
  --selection-dir "$OUT/selection" \
  --training-dir "$OUT/training" \
  --out-dir "$OUT/diagnostics"
