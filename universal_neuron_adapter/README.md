# Universal CLS-neuron adapter

This package implements one score-space adapter shared by LaGoVAD, DeSC, DSANet, and VadCLIP on UCF-Crime and XD-Violence. Every evaluation receives exactly one frozen baseline score stream. It never uses another baseline as an anchor. A neuron is one coordinate of a CLIP ViT-B/16 CLS hidden state; no optical flow or patch token is extracted.

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

## Official detection metric

Run only on the remote server in `conda activate dsanet`:

```bash
bash universal_neuron_adapter/commands/run_detection_map.sh
```

This executes the unchanged official DSANet detection-mAP function. Outputs remain below `../vadmy_data/universal_neuron_adapter`.
