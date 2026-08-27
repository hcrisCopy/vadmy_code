# Universal CLS-neuron adapter

This directory contains a single score-space adapter shared by LaGoVAD, DeSC, and DSANet on UCF-Crime and XD-Violence. Each evaluation accepts only the current baseline's score stream; scores from another baseline are never inputs. A neuron is exactly one coordinate of a CLIP ViT-B/16 CLS hidden state. The adapter uses no optical flow or patch tokens.

The current hypothesis combines the frozen baseline logit with three CLS-neuron experts and a multi-scale directional student. The student reads only the 384 selected CLS coordinates at the current snippet and two cheap temporal Gaussian scales; it never extracts new video features. A training-only persistence statistic produces one duration factor, which continuously controls contextual fusion, conservative correction, agreement residuals, normal-video suppression, and final temporal dilation. The formula and endpoint coefficients are identical for every baseline and dataset. All six evaluations use strict 16-frame snippet expansion and write only below `../vadmy_data/universal_neuron_adapter`.

Run remotely from the repository root with:

```bash
bash universal_neuron_adapter/commands/run_all.sh
```

