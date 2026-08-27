# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Each evaluation accepts only the current baseline's score stream; scores from another baseline are never inputs. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current hypothesis calibrates a baseline against the empirical CDF of its own training scores, expands short event responses with a fixed temporal maximum filter, and adds video-wise standardized CLS-neuron evidence in logit space. The same rank, expansion, and neuron coefficients are used for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

