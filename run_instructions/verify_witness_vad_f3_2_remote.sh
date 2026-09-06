#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash run_instructions/verify_witness_vad_f3_2_remote.sh 'ucf|xd|ucf xd'" >&2
  exit 2
fi

datasets="$1"
for dataset in $datasets; do
  if [[ "$dataset" != "ucf" && "$dataset" != "xd" ]]; then
    echo "datasets may contain only ucf and xd" >&2
    exit 2
  fi
done

out="../vadmy_data/witness_vad/dsanet/f3_2_signed_support"
data_root="$(realpath -m ../vadmy_data)"
out_abs="$(realpath -m "$out")"
expected_out="$data_root/witness_vad/dsanet/f3_2_signed_support"
receipt="run_instructions/retained_witness_vad_f3_2_xd.json"
metric="$out/target_margin.json"

if [[ "$out_abs" != "$expected_out" ]]; then
  echo "refusing unsafe F3.2 output path: $out_abs" >&2
  exit 2
fi

read_json_key() {
  python - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get(sys.argv[2], ""))
PY
}

config=""
if [[ "$datasets" != *" "* ]]; then
  config="$out/$datasets/w6/training/config.json"
fi
recorded_commit=""
if [[ -n "$config" && -f "$config" ]]; then
  recorded_commit="$(read_json_key "$config" git_commit)"
fi

method_matches() {
  [[ -n "$1" ]] && git cat-file -e "$1^{commit}" 2>/dev/null && \
    git diff --quiet "$1"..HEAD -- \
      vin_vad \
      run_instructions/run_witness_vad_f3_2_dsanet.sh
}

if [[ -f "$metric" ]] && method_matches "$recorded_commit"; then
  echo "reuse matching formal result from $recorded_commit"
  read_json_key "$metric" target_margin_pp
  exit 0
fi

if [[ -n "$config" && -f "$config" ]] && method_matches "$recorded_commit"; then
  echo "resume matching interrupted formal run from $recorded_commit"
  export WITNESS_DATASETS="$datasets"
  bash run_instructions/run_witness_vad_f3_2_dsanet.sh --resume
  exit 0
fi

# The archived controller log is the authoritative receipt for the completed
# a8b43cd XD run.  It prevents a documentation-only commit from retraining the
# exact same model after the controller record is archived.
if [[ "$datasets" == "xd" && -f "$receipt" ]]; then
  receipt_commit="$(read_json_key "$receipt" git_commit)"
  if method_matches "$receipt_commit"; then
    echo "reuse archived formal receipt from $receipt_commit"
    read_json_key "$receipt" target_margin_pp
    exit 0
  fi
fi

# Never delete a previous trial.  Move even an interrupted directory into the
# diagnostics archive before starting a clean, code-incompatible experiment.
if [[ -d "$out_abs" ]]; then
  archive_root="$data_root/witness_vad/dsanet/diagnostics/formal_trials"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  label="${recorded_commit:-incomplete}"
  archive_path="$archive_root/$stamp-${label:0:12}"
  mkdir -p "$archive_root"
  if [[ -e "$archive_path" ]]; then
    echo "refusing to overwrite formal trial archive: $archive_path" >&2
    exit 2
  fi
  mv -- "$out_abs" "$archive_path"
  echo "archived previous formal output: $archive_path"
fi

export WITNESS_DATASETS="$datasets"
bash run_instructions/run_witness_vad_f3_2_dsanet.sh --clean
