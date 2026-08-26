#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

RUN_KEY="$(git rev-parse --short HEAD)"
ROOT="../vadmy_data/universal_neuron_adapter"
OUT="$ROOT/runs/$RUN_KEY"
mkdir -p "$OUT"
printf '%s\n' "$RUN_KEY" > "$ROOT/current_run.txt"
exec > >(tee -a "$OUT/run.log") 2>&1

prepare_dataset() {
  local dataset="$1"
  local work train_csv test_csv train_hidden test_hidden missing
  if [[ "$dataset" == "ucf" ]]; then
    work="../vad_data/work_ucf"
    train_csv="$work/ucf_train_local.csv"
    test_csv="$work/ucf_test_local.csv"
    train_hidden="$work/clip_hidden_stride16_train_8gpu/manifest.csv"
    test_hidden="$work/clip_hidden_stride16_test_8gpu/manifest.csv"
    missing=()
  else
    work="../vad_data/work_xd"
    train_csv="$work/xd_train_local.csv"
    test_csv="$work/xd_test_local.csv"
    train_hidden="$work/clip_hidden_stride16_train_8gpu/manifest.csv"
    test_hidden="$work/clip_hidden_stride16_test_8gpu/manifest.csv"
    missing=(--skip-missing-hidden)
  fi
  local data_out="$OUT/$dataset/data"
  python -m universal_neuron_adapter.data \
    --dataset "$dataset" \
    --train-csv "$train_csv" \
    --test-csv "$test_csv" \
    --train-hidden-manifest "$train_hidden" \
    --test-hidden-manifest "$test_hidden" \
    --out-dir "$data_out" \
    --seed 234 \
    --val-fraction 0.2 \
    "${missing[@]}"

  local expert_out="$OUT/$dataset/expert"
  local resume=()
  if [[ -f "$expert_out/checkpoint_last.pth" ]]; then resume=(--resume); fi
  python -m universal_neuron_adapter.train_expert \
    --train-manifest "$data_out/expert_train.csv" \
    --val-manifest "$data_out/expert_val.csv" \
    --out-dir "$expert_out" \
    --active-per-layer 32 \
    --temporal-width 64 \
    --max-epoch 20 \
    --batch-size 8 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --sparsity-weight 1e-3 \
    --maximum-length 256 \
    --num-workers 4 \
    --seed 234 \
    --device cuda \
    "${resume[@]}"

  python -m universal_neuron_adapter.export_expert \
    --manifest "$data_out/train_all.csv" \
    --expert-model "$expert_out/expert_best.pth" \
    --out-dir "$expert_out/train" \
    --device cuda
  python -m universal_neuron_adapter.export_expert \
    --manifest "$data_out/test.csv" \
    --expert-model "$expert_out/expert_best.pth" \
    --out-dir "$expert_out/test" \
    --device cuda
}

run_baseline() {
  local dataset="$1"
  local baseline="$2"
  local data_out="$OUT/$dataset/data"
  local expert_out="$OUT/$dataset/expert"
  local baseline_root weight_args
  case "$baseline" in
    dsanet)
      baseline_root="baseline/DSANet"
      if [[ "$dataset" == "ucf" ]]; then
        weight_args=(--baseline-weight ../vadmy_data/model/DSANet/model_ucf.pth)
      else
        weight_args=(--baseline-weight ../vadmy_data/model/DSANet/model_xd.pth)
      fi
      ;;
    desc)
      baseline_root="baseline/DeSC"
      if [[ "$dataset" == "ucf" ]]; then
        weight_args=(--sensitivity-weight ../vadmy_data/model/DeSC/ucf_sensitivity_stream.pth --consistency-weight ../vadmy_data/model/DeSC/ucf_consistency_stream.pth)
      else
        weight_args=(--sensitivity-weight ../vadmy_data/model/DeSC/xd_sensitivity_stream.pth --consistency-weight ../vadmy_data/model/DeSC/xd_consistency_stream.pth)
      fi
      ;;
    lagovad)
      baseline_root="baseline/LaGoVAD-PreVAD"
      weight_args=(--baseline-weight ../vadmy_data/model/LaGoVAD/best.ckpt)
      ;;
    *)
      printf 'unsupported baseline: %s\n' "$baseline" >&2
      exit 2
      ;;
  esac
  local base_out="$OUT/$dataset/$baseline"
  python -m universal_neuron_adapter.cache_baseline \
    --baseline "$baseline" --baseline-root "$baseline_root" "${weight_args[@]}" \
    --dataset "$dataset" --manifest "$data_out/train_all.csv" --split train \
    --out-dir "$base_out/baseline_train" --device cuda
  python -m universal_neuron_adapter.cache_baseline \
    --baseline "$baseline" --baseline-root "$baseline_root" "${weight_args[@]}" \
    --dataset "$dataset" --manifest "$data_out/test.csv" --split test \
    --out-dir "$base_out/baseline_test" --device cuda

  local correction_out="$base_out/correction"
  local resume=()
  if [[ -f "$correction_out/checkpoint_last.pth" ]]; then resume=(--resume); fi
  python -m universal_neuron_adapter.train_correction \
    --baseline-manifest "$base_out/baseline_train/baseline_scores.csv" \
    --expert-manifest "$expert_out/train/expert_scores.csv" \
    --train-keys "$data_out/expert_train.csv" \
    --val-keys "$data_out/expert_val.csv" \
    --baseline "$baseline" --dataset "$dataset" \
    --out-dir "$correction_out" \
    --width 32 --max-epoch 15 --batch-size 32 --lr 3e-4 --weight-decay 1e-4 \
    --maximum-length 256 --num-workers 4 --seed 234 --device cuda \
    "${resume[@]}"
  python -m universal_neuron_adapter.evaluate \
    --baseline-manifest "$base_out/baseline_test/baseline_scores.csv" \
    --expert-manifest "$expert_out/test/expert_scores.csv" \
    --correction-model "$correction_out/model_best.pth" \
    --gt-path "../vadmy_data/annotations/$dataset/gt.npy" \
    --baseline "$baseline" --dataset "$dataset" \
    --out-dir "$base_out/evaluation" \
    --frames-per-snippet 16 --device cuda
}

for dataset in ucf xd; do
  prepare_dataset "$dataset"
  for baseline in lagovad desc dsanet; do
    run_baseline "$dataset" "$baseline"
  done
done

python -m universal_neuron_adapter.aggregate_metric --results-root "$ROOT"

