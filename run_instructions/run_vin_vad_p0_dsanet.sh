#!/usr/bin/env bash
set -euo pipefail

OUT=../vadmy_data/vin_vad/dsanet/p0
CLEAN_ARGS=()
if [[ "${CLEAN:-0}" == "1" ]]; then
  CLEAN_ARGS=(--clean)
fi

python -m pytest vin_vad/tests/test_p0.py -q

python -m vin_vad.p0_audit \
  --dataset ucf \
  --train-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --test-csv ../vad_data/work_ucf/ucf_test_local.csv \
  --train-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_ucf/clip_hidden_stride16_test_8gpu/manifest.csv \
  --gt-path baseline/DSANet/list/gt_ucf.npy \
  --out-dir "$OUT/ucf" \
  --frames-per-snippet 16 \
  "${CLEAN_ARGS[@]}"

python -m vin_vad.p0_audit \
  --dataset xd \
  --train-csv ../vad_data/work_xd/xd_train_local.csv \
  --test-csv ../vad_data/work_xd/xd_test_local.csv \
  --train-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_train_8gpu/manifest.csv \
  --test-hidden-manifest ../vad_data/work_xd/clip_hidden_stride16_test_8gpu/manifest.csv \
  --gt-path baseline/DSANet/list/gt.npy \
  --out-dir "$OUT/xd" \
  --frames-per-snippet 16 \
  --skip-missing-train-hidden \
  "${CLEAN_ARGS[@]}"

python - <<'PY'
import json
from pathlib import Path

for dataset in ("ucf", "xd"):
    path = Path("../vadmy_data/vin_vad/dsanet/p0") / dataset / "audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{dataset}: {audit['status']}; videos={audit['train_videos']}+{audit['test_videos']}; "
        f"GT frames={audit['test_gt_frames']}; dropped test tails={audit['test_tail_snippets_dropped']}"
    )
PY
