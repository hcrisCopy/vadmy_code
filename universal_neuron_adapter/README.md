# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current research hypothesis first distills the train-only DeSC/DSANet score consensus into one sparse CLS-neuron expert per dataset. At test time the peer baselines are absent: each target baseline is combined only with its own score and the shared neuron expert. The same sparse architecture, losses, and fusion coefficients (`0.2` and `0.1`) are used for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```
