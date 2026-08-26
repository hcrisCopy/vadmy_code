# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current hypothesis is a neuron-gated dual-anchor consensus. DeSC and DSANet provide complementary score anchors. Their raw probabilities and empirical percentiles are averaged, where the empirical CDF is fitted exclusively on training scores. A fixed temporal maximum filter expands event evidence, and video-wise standardized CLS-neuron evidence adjusts the final logit. The complete parameter set is identical for every baseline and dataset.

This variant requires DeSC and DSANet inference scores at test time, even when reporting the LaGoVAD-adapted result. It adds no optical flow, patch-token extraction, or backbone feature extraction beyond the provided baseline features and pre-extracted CLS hidden states. Baselines can be run sequentially on one RTX 4090.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

