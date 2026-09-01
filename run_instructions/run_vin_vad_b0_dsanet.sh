#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/b0
SEED=42
export OMP_NUM_THREADS=8
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "B0 DSANet host identity; seed=${SEED}"
python -m pytest vin_vad/tests/test_p0.py vin_vad/tests/test_b0.py -q

for dataset in ucf xd; do
  if [[ "$dataset" == "ucf" ]]; then
    train_csv=../vad_data/work_ucf/ucf_train_local.csv
    test_csv=../vad_data/work_ucf/ucf_test_local.csv
    train_hidden=../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv
    test_hidden=../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv
    gt=baseline/DSANet/list/gt_ucf.npy
    weight=../vadmy_data/model/DSANet/model_ucf.pth
    missing_args=()
  else
    train_csv=../vad_data/work_xd/xd_train_local.csv
    test_csv=../vad_data/work_xd/xd_test_local.csv
    train_hidden=../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv
    test_hidden=../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv
    gt=baseline/DSANet/list/gt.npy
    weight=../vadmy_data/model/DSANet/model_xd.pth
    missing_args=(--skip-missing-train-hidden)
  fi

  data_dir="$OUT/$dataset/data"
  host_dir="$OUT/$dataset/host"
  evaluation_dir="$OUT/$dataset/evaluation"

  python -m vin_vad.p0_audit \
    --dataset "$dataset" \
    --train-csv "$train_csv" \
    --test-csv "$test_csv" \
    --train-hidden-manifest "$train_hidden" \
    --test-hidden-manifest "$test_hidden" \
    --gt-path "$gt" \
    --out-dir "$data_dir" \
    --frames-per-snippet 16 \
    "${missing_args[@]}" \
    "${CLEAN_ARGS[@]}"

  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet \
    --baseline-root baseline/DSANet \
    --baseline-weight "$weight" \
    --dataset "$dataset" \
    --manifest "$data_dir/train.csv" \
    --split train \
    --out-dir "$host_dir/train" \
    --device cuda \
    --resume \
    "${CLEAN_ARGS[@]}"

  python -m universal_neuron_adapter.cache_baseline \
    --baseline dsanet \
    --baseline-root baseline/DSANet \
    --baseline-weight "$weight" \
    --dataset "$dataset" \
    --manifest "$data_dir/test.csv" \
    --split test \
    --out-dir "$host_dir/test" \
    --device cuda \
    --resume \
    "${CLEAN_ARGS[@]}"

  python -m vin_vad.b0_identity \
    --dataset "$dataset" \
    --train-manifest "$data_dir/train.csv" \
    --test-manifest "$data_dir/test.csv" \
    --train-host-manifest "$host_dir/train/baseline_scores.csv" \
    --test-host-manifest "$host_dir/test/baseline_scores.csv" \
    --gt-path "$gt" \
    --out-dir "$evaluation_dir" \
    --target-tpr 0.95 \
    "${CLEAN_ARGS[@]}"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("../vadmy_data/vin_vad/dsanet/b0")
for dataset in ("ucf", "xd"):
    metrics = json.loads((root / dataset / "evaluation" / "metrics.json").read_text())
    primary = "pooled_auc" if dataset == "ucf" else "pooled_ap"
    print(
        f"{dataset}: {primary}={100 * metrics[primary]:.3f}, "
        f"Cross-AUC={100 * metrics['cross_auc']:.3f}, "
        f"Macro-Within-AUC={100 * metrics['macro_within_auc']:.3f}, "
        f"identity-error={metrics['host_identity_max_abs_error']:.1e}"
    )
PY
