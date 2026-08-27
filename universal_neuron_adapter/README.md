# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current retained hypothesis conservatively combines the frozen baseline logit, the first-round learned neuron-correction logit, and video-wise standardized neuron evidence. The same coefficients (`0.2` and `0.1`) are used for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

