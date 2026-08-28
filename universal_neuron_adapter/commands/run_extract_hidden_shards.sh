#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_extract_hidden_shards.sh <shards_dir> <video_root> <out_root> [num_gpus=8] [stride=16] [batch_size=128] [layers=all] [drop_last=0]" >&2
  exit 2
fi
SHARDS_DIR="$1"
VIDEO_ROOT="$2"
OUT_ROOT="$3"
NUM_GPUS="${4:-8}"
STRIDE="${5:-16}"
BATCH_SIZE="${6:-128}"
LAYERS="${7:-all}"
DROP_LAST="${8:-0}"
mkdir -p "$OUT_ROOT/logs"
PIDS=()
for SID in $(seq 0 $((NUM_GPUS-1))); do
  CSV_PATH="$SHARDS_DIR/shard_${SID}.csv"
  if [ ! -f "$CSV_PATH" ]; then
    echo "missing shard csv: $CSV_PATH" >&2
    exit 1
  fi
  CMD=(python -m universal_neuron_adapter.extract_hidden_states
    --dsanet-root baseline/DSANet
    --video-root "$VIDEO_ROOT"
    --out-dir "$OUT_ROOT/shard_${SID}"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --layers "$LAYERS"
    --filter-csv "$CSV_PATH")
  if [ "$DROP_LAST" = "1" ] || [ "$DROP_LAST" = "true" ] || [ "$DROP_LAST" = "yes" ]; then
    CMD+=(--drop-last-incomplete)
  fi
  if [ "$SID" = "0" ]; then
    echo "started hidden shard 0 on GPU 0; showing its progress below"
    CUDA_VISIBLE_DEVICES="$SID" "${CMD[@]}" 2>&1 | tee "$OUT_ROOT/logs/shard_${SID}.log" &
  else
    CUDA_VISIBLE_DEVICES="$SID" "${CMD[@]}" > "$OUT_ROOT/logs/shard_${SID}.log" 2>&1 &
    echo "started hidden shard $SID on GPU $SID, log=$OUT_ROOT/logs/shard_${SID}.log"
  fi
  PIDS+=("$!")
done
FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done
if [ "$FAIL" != "0" ]; then
  echo "at least one shard failed; check logs under $OUT_ROOT/logs" >&2
  exit 1
fi
python -m universal_neuron_adapter.merge_manifests --input-root "$OUT_ROOT" --output-csv "$OUT_ROOT/manifest.csv" --kind manifest
