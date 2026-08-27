# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Each evaluation accepts only the current baseline's score stream; scores from another baseline are never inputs. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current hypothesis conservatively combines the frozen baseline logit, the first-round learned neuron-correction logit, and video-wise standardized neuron evidence. It expands event peaks only where the same CLS-neuron evidence opens a soft gate. A current-baseline training classifier then suppresses only high-confidence normal videos; it never boosts a suspected abnormal video. The same features, model, and coefficients are used for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

