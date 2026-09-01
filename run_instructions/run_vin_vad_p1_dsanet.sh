#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
OUT=../vadmy_data/vin_vad/dsanet/p1
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

python -m pytest \
  vin_vad/tests/test_p0.py \
  vin_vad/tests/test_event_chain.py \
  vin_vad/tests/test_base_tcn.py -q

python -m vin_vad.prepare_p1 \
  --train-manifest ../vadmy_data/vin_vad/dsanet/p0/ucf/train.csv \
  --test-manifest ../vadmy_data/vin_vad/dsanet/p0/ucf/test.csv \
  --out-dir "$OUT/ucf/data" \
  "${CLEAN_ARGS[@]}"

python -m vin_vad.prepare_p1 \
  --train-manifest ../vadmy_data/vin_vad/dsanet/p0/xd/train.csv \
  --test-manifest ../vadmy_data/vin_vad/dsanet/p0/xd/test.csv \
  --out-dir "$OUT/xd/data" \
  "${CLEAN_ARGS[@]}"

python -m vin_vad.fixed_emission --out-path "$OUT/fixed_emission.json"

run_ablation() {
  local dataset="$1"
  local variant="$2"
  local batch_size="$3"
  local learning_rate="$4"
  local ground_truth="$5"
  local run_dir="$OUT/$dataset/$variant"

  python -m vin_vad.train \
    --manifest "$OUT/$dataset/data/train.csv" \
    --variant "$variant" \
    --out-dir "$run_dir" \
    --epochs 10 \
    --batch-size "$batch_size" \
    --lr "$learning_rate" \
    --weight-decay 0.01 \
    --maximum-length 256 \
    --width 512 \
    --dropout 0.1 \
    --num-workers 4 \
    --seed 234 \
    --device cuda \
    "${CLEAN_ARGS[@]}"

  python -m vin_vad.evaluate \
    --manifest "$OUT/$dataset/data/test.csv" \
    --checkpoint "$run_dir/model_final.pt" \
    --gt-path "$ground_truth" \
    --variant "$variant" \
    --dataset "$dataset" \
    --out-dir "$run_dir/evaluation" \
    --width 512 \
    --dropout 0.1 \
    --num-workers 4 \
    --device cuda
}

for variant in e0 e1 e2 e3; do
  run_ablation ucf "$variant" 64 0.00007 baseline/DSANet/list/gt_ucf.npy
done

for variant in e0 e1 e2 e3; do
  run_ablation xd "$variant" 96 0.00001 baseline/DSANet/list/gt.npy
done

python -m vin_vad.summarize_p1 --root "$OUT"
printf 'P1 complete: %s\n' "$OUT"
