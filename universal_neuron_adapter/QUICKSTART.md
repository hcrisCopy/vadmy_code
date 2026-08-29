# Quick start

Run every command on the remote server from `../vadmy_code` after `conda activate dsanet`. Inputs under `../vad_data` are read-only; every new artifact is written under `../vadmy_data`.

## 1. Extract CLS hidden states when needed

```bash
python -m universal_neuron_adapter.make_csv_shards \
  --input-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --out-dir ../vadmy_data/hidden_extract/ucf_train_shards --num-shards 1

bash universal_neuron_adapter/commands/run_extract_hidden_shards.sh \
  ../vadmy_data/hidden_extract/ucf_train_shards <video_root> \
  ../vadmy_data/hidden_extract/ucf_train 1 16 128 all 0
```

Output: `[T,12,768]` float16 CLS hidden states and manifests under `../vadmy_data/hidden_extract`.

## 2. Run the six formal evaluations

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

Output: per-video curves and metrics under `../vadmy_data/universal_neuron_adapter/runs/<git-short-sha>`, plus the six-way summary in `summary.json`.

## 3. Run checks and figures

```bash
python -m universal_neuron_adapter.validate_constraints
python -m pip install -r universal_neuron_adapter/requirements-dev.txt
python -m pytest -q universal_neuron_adapter/tests
bash universal_neuron_adapter/commands/run_neuron_visualization.sh --clean
```

Output: constraint/test reports in the terminal and figures under `../vadmy_data/universal_neuron_adapter/figures/detected_neurons`.

The split audit must report zero official train/test key overlap, zero hidden-state overlap, and zero validation/test overlap before a result is used.
