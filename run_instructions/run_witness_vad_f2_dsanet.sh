#!/usr/bin/env bash
set -euo pipefail

OUT="../vadmy_data/witness_vad/dsanet/f2_train_contract"
DATA_ROOT="$(realpath -m ../vadmy_data)"
OUT_ABS="$(realpath -m "$OUT")"
EXPECTED_OUT="$DATA_ROOT/witness_vad/dsanet/f2_train_contract"

if [[ "$OUT_ABS" != "$EXPECTED_OUT" ]]; then
  echo "refusing unsafe F2 output path: $OUT_ABS" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" && "$1" != "--clean" ) ]]; then
  echo "usage: bash run_instructions/run_witness_vad_f2_dsanet.sh [--resume|--clean]" >&2
  exit 2
fi
if [[ "${1:-}" == "--clean" && -d "$OUT_ABS" ]]; then
  echo "cleaning exact F2 output: $OUT_ABS"
  rm -rf -- "$OUT_ABS"
fi

mkdir -p "$OUT"
COMMAND="bash run_instructions/run_witness_vad_f2_dsanet.sh${1:+ $1}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
exec > >(tee -a "$OUT/stdout.log") 2>&1

echo "F2 Witness-VAD training, resume and determinism contract"
echo "output: $OUT_ABS"
python -m pytest \
  vin_vad/tests/test_witness_neurons.py \
  vin_vad/tests/test_witness_router.py \
  vin_vad/tests/test_witness_losses.py \
  vin_vad/tests/test_train_witness.py \
  -q | tee "$OUT/test_report.txt"

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_ARGS=(--resume)
fi

python -m vin_vad.train_witness \
  --dataset ucf \
  --train-manifest ../vadmy_data/vin_vad/dsanet/b0/ucf/evaluation/train_aligned.csv \
  --out-dir "$OUT/reference" \
  --epochs 2 \
  --stop-after-epoch 0 \
  --batch-size 8 \
  --maximum-length 256 \
  --videos-per-class 8 \
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

python -m vin_vad.train_witness \
  --dataset ucf \
  --train-manifest ../vadmy_data/vin_vad/dsanet/b0/ucf/evaluation/train_aligned.csv \
  --out-dir "$OUT/resumed" \
  --epochs 2 \
  --stop-after-epoch 1 \
  --batch-size 8 \
  --maximum-length 256 \
  --videos-per-class 8 \
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

if [[ ! -f "$OUT/interruption_metrics.json" ]]; then
  cp "$OUT/resumed/metrics.json" "$OUT/interruption_metrics.json"
fi

python -m vin_vad.train_witness \
  --dataset ucf \
  --train-manifest ../vadmy_data/vin_vad/dsanet/b0/ucf/evaluation/train_aligned.csv \
  --out-dir "$OUT/resumed" \
  --epochs 2 \
  --stop-after-epoch 0 \
  --batch-size 8 \
  --maximum-length 256 \
  --videos-per-class 8 \
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
  --resume

python -m vin_vad.audit_f2 \
  --reference-dir "$OUT/reference" \
  --resumed-dir "$OUT/resumed" \
  --interruption-record "$OUT/interruption_metrics.json" \
  --out-dir "$OUT" \
  --relative-loss-tolerance 0.01 \
  --parameter-tolerance 0.0000001

echo "F2 complete: $OUT/summary.md"
