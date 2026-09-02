#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

OUT="../vadmy_data/witness_vad/dsanet/f3_eta_probe"
F3="../vadmy_data/witness_vad/dsanet/f3_performance"
DATA_ROOT="$(realpath -m ../vadmy_data)"
OUT_ABS="$(realpath -m "$OUT")"
EXPECTED_OUT="$DATA_ROOT/witness_vad/dsanet/f3_eta_probe"

if [[ "$OUT_ABS" != "$EXPECTED_OUT" ]]; then
  echo "refusing unsafe F3.1 output path: $OUT_ABS" >&2
  exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" && "$1" != "--clean" ) ]]; then
  echo "usage: bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh [--resume|--clean]" >&2
  exit 2
fi
if [[ ! -f "$F3/ucf/w2/training/checkpoints/best.pt" || ! -f "$F3/xd/w6/training/checkpoints/best.pt" ]]; then
  echo "missing F3 best checkpoints under $F3" >&2
  exit 2
fi
if [[ "${1:-}" == "--clean" && -d "$OUT_ABS" ]]; then
  echo "cleaning exact F3.1 output: $OUT_ABS"
  rm -rf -- "$OUT_ABS"
fi

mkdir -p "$OUT"
COMMAND="bash run_instructions/run_witness_vad_f3_eta_probe_dsanet.sh${1:+ $1}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
exec > >(tee -a "$OUT/stdout.log") 2>&1

echo "F3.1 Witness-VAD DSANet inference-only eta_A probe"
echo "output: $OUT_ABS"
python -m pytest \
  vin_vad/tests/test_witness_router.py \
  vin_vad/tests/test_witness_variants.py \
  -q | tee "$OUT/test_report.txt"

for dataset in ucf xd; do
  if [[ "$dataset" == "ucf" ]]; then
    ground_truth="baseline/DSANet/list/gt_ucf.npy"
  else
    ground_truth="baseline/DSANet/list/gt.npy"
  fi
  test_manifest="../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/test_aligned.csv"
  host_metrics="../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/metrics.json"
  for variant in w2 w6; do
    checkpoint="$F3/$dataset/$variant/training/checkpoints/best.pt"
    for eta in 0.25 0.35 0.60; do
      eta_tag="eta_${eta/./_}"
      python -m vin_vad.evaluate_witness \
        --dataset "$dataset" \
        --variant "$variant" \
        --test-manifest "$test_manifest" \
        --checkpoint "$checkpoint" \
        --host-metrics "$host_metrics" \
        --gt-path "$ground_truth" \
        --out-dir "$OUT/$dataset/$variant/$eta_tag" \
        --target-tpr 0.95 \
        --device cuda \
        --selection-policy test_primary_metric_best \
        --eta-anomaly-override "$eta"
    done
  done
done

python -m vin_vad.summarize_witness_eta_probe --root "$OUT" --f3-root "$F3"
echo "F3.1 complete: $OUT/summary.md"
