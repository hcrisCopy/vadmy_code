#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

OUT="../vadmy_data/witness_vad/dsanet/f3_performance"
DATA_ROOT="$(realpath -m ../vadmy_data)"
OUT_ABS="$(realpath -m "$OUT")"
EXPECTED_OUT="$DATA_ROOT/witness_vad/dsanet/f3_performance"

if [[ "$OUT_ABS" != "$EXPECTED_OUT" ]]; then
  echo "refusing unsafe F3 output path: $OUT_ABS" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" && "$1" != "--clean" ) ]]; then
  echo "usage: bash run_instructions/run_witness_vad_f3_dsanet.sh [--resume|--clean]" >&2
  exit 2
fi
if [[ "${1:-}" == "--clean" && -d "$OUT_ABS" ]]; then
  echo "cleaning exact F3 output: $OUT_ABS"
  rm -rf -- "$OUT_ABS"
fi

mkdir -p "$OUT"
COMMAND="bash run_instructions/run_witness_vad_f3_dsanet.sh${1:+ $1}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
exec > >(tee -a "$OUT/stdout.log") 2>&1

echo "F3 Witness-VAD DSANet performance gate"
echo "output: $OUT_ABS"
python -m pytest \
  vin_vad/tests/test_witness_neurons.py \
  vin_vad/tests/test_witness_router.py \
  vin_vad/tests/test_witness_losses.py \
  vin_vad/tests/test_witness_variants.py \
  vin_vad/tests/test_train_witness.py \
  -q | tee "$OUT/test_report.txt"

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_ARGS=(--resume)
fi

for dataset in ucf xd; do
  if [[ "$dataset" == "ucf" ]]; then
    ground_truth="baseline/DSANet/list/gt_ucf.npy"
  else
    ground_truth="baseline/DSANet/list/gt.npy"
  fi
  test_manifest="../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/test_aligned.csv"
  train_manifest="../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/train_aligned.csv"
  host_metrics="../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/metrics.json"

  python -m vin_vad.evaluate_witness \
    --dataset "$dataset" \
    --variant w0 \
    --test-manifest "$test_manifest" \
    --host-metrics "$host_metrics" \
    --gt-path "$ground_truth" \
    --out-dir "$OUT/$dataset/w0/evaluation" \
    --target-tpr 0.95 \
    --device cuda

  for variant in w1 w2 w6; do
    python -m vin_vad.train_witness \
      --dataset "$dataset" \
      --variant "$variant" \
      --train-manifest "$train_manifest" \
      --out-dir "$OUT/$dataset/$variant/training" \
      --epochs 20 \
      --stop-after-epoch 0 \
      --batch-size 8 \
      --maximum-length 256 \
      --videos-per-class 0 \
      --num-workers 4 \
      --active-neurons 32 \
      --temporal-width 64 \
      --eta-normal 1.0 \
      --eta-anomaly 0.25 \
      --lambda-witness 1.0 \
      --lambda-final 1.0 \
      --lambda-normal 0.5 \
      --lambda-sparse 0.001 \
      --rank-margin 0.5 \
      --rank-weight 0.5 \
      --smooth-weight 0.02 \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --seed 42 \
      --device cuda \
      "${RESUME_ARGS[@]}"

    python -m vin_vad.evaluate_witness \
      --dataset "$dataset" \
      --variant "$variant" \
      --test-manifest "$test_manifest" \
      --checkpoint "$OUT/$dataset/$variant/training/checkpoints/last.pt" \
      --host-metrics "$host_metrics" \
      --gt-path "$ground_truth" \
      --out-dir "$OUT/$dataset/$variant/evaluation" \
      --target-tpr 0.95 \
      --device cuda
  done
done

python -m vin_vad.summarize_f3 --root "$OUT"
echo "F3 complete: $OUT/summary.md"
