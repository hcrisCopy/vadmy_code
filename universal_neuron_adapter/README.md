# Universal CLS-neuron adapter

This package implements one score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Every evaluation receives exactly one frozen baseline score stream. It never uses another baseline as an anchor. A neuron is one coordinate of a CLIP ViT-B/16 CLS hidden state; no optical flow or patch token is extracted.

## Method

The adapter uses three distinct CLS-neuron views discovered separately for each dataset from official training videos:

1. **Primary sparse detector:** a Top-32-per-layer MIL detector that learns which CLS coordinates respond to abnormal videos.
2. **Multi-scale context detector:** a linear student over directional coordinates at the current snippet and two Gaussian temporal scales. It models local event context; it is not a wider copy of the primary detector.
3. **Directional normality detector:** a deterministic Top-32-per-layer detector fitted from normal training statistics. It keeps the abnormal direction of each selected coordinate and rejects the irrelevant tail.

Their positive response-correlation matrix defines a spectral consensus module: the principal eigenvector supplies mean-one detector weights, emphasizing views that agree and reducing isolated responses. Positive agreement supplies a bounded residual, joint negative evidence suppresses likely false positives, and training-only persistence controls temporal aggregation and boundary recovery. Every stochastic final component uses seed 234. No module consults another baseline stream.

## Data-integrity policy

`python -m universal_neuron_adapter.data` builds the validation split only from the official training list. Before writing a manifest it checks that:

- official training and test video keys are disjoint;
- training and test hidden-state keys are disjoint;
- validation and test keys are disjoint.

The four source manifests are fixed by SHA-256 in `prepare_signature.json`; the result is recorded in `split_audit.json`. A non-empty overlap terminates the run. Test labels are used only for final reporting, ablations, and explicitly labeled post-hoc analysis—not for gradient updates or checkpoint selection.

The formal numbers are nevertheless **development-test results**: the method structure was iterated using repeated benchmark feedback. This is not sample leakage, but it is test-aware model development. Do not describe the reported table as a blind held-out estimate. For confirmatory claims, freeze this implementation and evaluate on a newly held-out dataset or split.

## Hidden-state extraction

First create video-grouped CSV shards, then run extraction on the remote server:

```bash
python -m universal_neuron_adapter.make_csv_shards \
  --input-csv ../vad_data/work_ucf/ucf_train_local.csv \
  --out-dir ../vadmy_data/hidden_extract/ucf_train_shards --num-shards 1

bash universal_neuron_adapter/commands/run_extract_hidden_shards.sh \
  ../vadmy_data/hidden_extract/ucf_train_shards <video_root> \
  ../vadmy_data/hidden_extract/ucf_train 1 16 128 all 0
```

The extractor reproduces the existing `[T, 12, 768]` float16 files by walking the 12 CLIP ViT-B/16 visual blocks and saving each block's CLS token. Outputs must remain under `../vadmy_data`; existing files under `../vad_data` are read-only inputs.

## Supplementary experiments

Run only on the remote server in `conda activate dsanet`:

```bash
bash universal_neuron_adapter/commands/run_supplementary_experiments.sh
bash universal_neuron_adapter/commands/run_detection_map.sh
bash universal_neuron_adapter/commands/run_seed_study.sh
bash universal_neuron_adapter/commands/run_neuron_controls.sh
bash universal_neuron_adapter/commands/run_robustness_efficiency.sh
bash universal_neuron_adapter/commands/run_neuron_visualization.sh
```

The supplementary runner performs the split audit, a cumulative component ablation across all six settings, a 200-repeat exact paired video bootstrap over the official frame metric, and focused heatmap/timeline figures. The detection runner executes the official DSANet detection-mAP function and preserves DSANet's conditional abnormal-class distribution when replacing total anomaly mass. The seed study retrains all stochastic adapter components for seeds 234, 3407, and 2026. The neuron controls compare removal of selected neurons with five size-matched random removals.

The robustness runner perturbs event width (33/41/49) and score advance (0/1/2) one factor at a time. These test-set measurements are post-hoc robustness checks only and never select the formal setting. It also records cached-score adapter wall time, throughput, and peak CUDA allocation, then renders one focused gain heatmap.

All results, logs, checkpoints, cached curves, figures, and reports are written below `../vadmy_data/universal_neuron_adapter`.

The neuron-visualization command renders a paper-ready interpretability figure and one anomaly-response heatmap per dataset. Each dataset figure separates the primary sparse, multi-scale context, and directional normality views, ordered by training-only abnormal-versus-normal response effect. An exact 12-layer by 768-dimension coordinate heatmap is retained for supplementary inspection. PNG/PDF figures, source CSV tables, self-contained captions, and seed/data-provenance metadata are written under `../vadmy_data/universal_neuron_adapter/figures/detected_neurons`. Panel (c) uses training videos only; panel (d) is an explicitly labeled post-hoc neuron-removal control.
