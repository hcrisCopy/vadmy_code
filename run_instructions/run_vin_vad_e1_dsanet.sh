#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/e1
export OMP_NUM_THREADS=8
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
  rm -f "$OUT/run.log"
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest \
  vin_vad/tests/test_context_predictor.py \
  vin_vad/tests/test_violation_field.py \
  vin_vad/tests/test_host_auditor.py \
  vin_vad/tests/test_b4.py \
  vin_vad/tests/test_e1.py -q

for dataset in ucf xd; do
  if [[ "$dataset" == "ucf" ]]; then
    ground_truth=baseline/DSANet/list/gt_ucf.npy
  else
    ground_truth=baseline/DSANet/list/gt.npy
  fi
  for evidence in c0 c1 c2 c3 c4; do
    variant_dir="$OUT/$dataset/$evidence"
    python -m vin_vad.train \
      --stage e1 \
      --evidence "$evidence" \
      --dataset "$dataset" \
      --train-manifest "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/train_aligned.csv" \
      --b1-dir "../vadmy_data/vin_vad/dsanet/b1/$dataset" \
      --b2-checkpoint "../vadmy_data/vin_vad/dsanet/b2/$dataset/violation_field_initial.pt" \
      --global-statistics "../vadmy_data/vin_vad/dsanet/b1/$dataset/global_normal_statistics.npz" \
      --out-dir "$variant_dir/training" \
      --epochs 10 \
      --batch-size 8 \
      --lr 0.0003 \
      --weight-decay 0.0001 \
      --maximum-length 256 \
      --delta 1.0 \
      --statistics-momentum 0.05 \
      --alpha-cross 0.5 \
      --alpha-within 0.25 \
      --correction-budget 0.1 \
      --lambda-context 1.0 \
      --lambda-budget 10.0 \
      --q-reservoir-capacity 4096 \
      --normal-quantile 0.95 \
      --gradient-clip 5.0 \
      --num-workers 2 \
      --seed 42 \
      --device cuda \
      --resume \
      "${CLEAN_ARGS[@]}"

    python -m vin_vad.evaluate_correction \
      --dataset "$dataset" \
      --test-manifest "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/test_aligned.csv" \
      --checkpoint "$variant_dir/training/model_final.pt" \
      --host-metrics "../vadmy_data/vin_vad/dsanet/b0/$dataset/evaluation/metrics.json" \
      --gt-path "$ground_truth" \
      --out-dir "$variant_dir/evaluation" \
      --maximum-length 256 \
      --window-overlap 64 \
      --target-tpr 0.95 \
      --device cuda \
      "${CLEAN_ARGS[@]}"
  done

  python -m vin_vad.audit_context_replacement \
    --dataset "$dataset" \
    --validation-manifest "../vadmy_data/vin_vad/dsanet/b1/$dataset/data/validation_normal.csv" \
    --checkpoint "$OUT/$dataset/c3/training/model_final.pt" \
    --out-dir "$OUT/$dataset/c3/context_replacement" \
    --pairs 32 \
    --maximum-length 256 \
    --seed 42 \
    --device cuda \
    "${CLEAN_ARGS[@]}"
done

python -m vin_vad.summarize_e1 --root "$OUT"
