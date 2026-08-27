# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Each evaluation accepts only the current baseline's score stream; scores from another baseline are never inputs. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current hypothesis conservatively combines the frozen baseline logit, the first-round learned neuron-correction logit, and two sparse CLS-neuron experts. Experts with 32 and 64 active dimensions per CLIP layer jointly gate event propagation. A joint current-baseline/CLS classifier suppresses only high-confidence normal videos, a fixed median filter removes isolated spikes, and scores are advanced one snippet within each video. The same features, models, and coefficients are used for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

