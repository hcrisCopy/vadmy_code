# Universal CLS-neuron adapter

This package implements one score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Every evaluation receives exactly one frozen baseline score stream. It never uses another baseline as an anchor. A neuron is one coordinate of a CLIP ViT-B/16 CLS hidden state; no optical flow or patch token is extracted.

## Method

The adapter combines a conservative learned score correction with three sparse CLS-neuron views. Agreement gates permit local event expansion only where baseline and neuron evidence support an anomaly. A one-sided video prior suppresses likely normal videos. The temporal width is inferred from training-video evidence and controls persistence smoothing and dilation through one shared continuous rule. The formula and coefficients are identical for all six baseline/dataset settings.

## Data-integrity policy

`python -m universal_neuron_adapter.data` builds the validation split only from the official training list. Before writing a manifest it checks that:

- official training and test video keys are disjoint;
- training and test hidden-state keys are disjoint;
- validation and test keys are disjoint.

The four source manifests are fixed by SHA-256 in `prepare_signature.json`; the result is recorded in `split_audit.json`. A non-empty overlap terminates the run. Test labels are used only for final reporting, ablations, and explicitly labeled post-hoc analysis—not for gradient updates or checkpoint selection.

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
```

The supplementary runner performs the split audit, a cumulative component ablation across all six settings, a 2,000-repeat paired video bootstrap, and focused heatmap/timeline figures. The detection runner executes the official DSANet detection-mAP function and preserves DSANet's conditional abnormal-class distribution when replacing total anomaly mass. The seed study retrains all stochastic adapter components for seeds 234, 3407, and 2026. The neuron controls compare removal of selected neurons with five size-matched random removals.

All results, logs, checkpoints, cached curves, figures, and reports are written below `../vadmy_data/universal_neuron_adapter`.
