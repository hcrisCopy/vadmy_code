# Research log

## Trial 1 — sparse CLS-neuron expert and learned score correction (`9d1a066`, discarded)

The shared sparse expert selected 32 CLS hidden-state dimensions per CLIP layer and trained a small temporal expert. Its held-out video discrimination was strong (about 0.978 AUC/AP on UCF and 0.992 on XD), but the video-level correction objective did not reliably improve frame ranking. Paper-relative gains were: LaGoVAD UCF +4.857, LaGoVAD XD +3.185, DeSC UCF +0.002, DeSC XD -0.091, DSANet UCF +0.091, and DSANet XD +0.252 pp. Minimum gain: -0.091 pp. The failure indicates that unrestricted correction can overfit bag discrimination and perturb an already strong frame ordering.

## Trial 2 — conservative logit shrinkage plus standardized neuron evidence

Hypothesis: retain only 20% of the learned correction and add 0.1 times video-wise standardized neuron evidence in logit space. Standardization makes the identical coefficient meaningful across all baselines and datasets; shrinkage protects strong baseline rankings. Formal results were LaGoVAD UCF 82.960 (+1.840), LaGoVAD XD 76.627 (+2.377), DeSC UCF 89.437 (+0.067), DeSC XD 87.366 (+0.186), DSANet UCF 89.482 (+0.042), and DSANet XD 87.476 (+0.526). Minimum gain: +0.042 pp. The trial was retained.

## Trial 3 — train-only consensus distillation into sparse CLS neurons

Hypothesis: the original video-MIL neuron expert lacks temporal localization. Use the average DeSC/DSANet train logits as dense soft targets for abnormal videos and zero targets for known-normal videos, then distill them into the same 32-neurons-per-layer expert. Other baselines are used only to create training supervision; test inference remains target-baseline plus CLS-neuron expert. Formal verification is pending.
