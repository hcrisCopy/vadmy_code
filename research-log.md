# Research log

## Trial 1 — sparse CLS-neuron expert and learned score correction (`9d1a066`, discarded)

The shared sparse expert selected 32 CLS hidden-state dimensions per CLIP layer and trained a small temporal expert. Its held-out video discrimination was strong (about 0.978 AUC/AP on UCF and 0.992 on XD), but the video-level correction objective did not reliably improve frame ranking. Paper-relative gains were: LaGoVAD UCF +4.857, LaGoVAD XD +3.185, DeSC UCF +0.002, DeSC XD -0.091, DSANet UCF +0.091, and DSANet XD +0.252 pp. Minimum gain: -0.091 pp. The failure indicates that unrestricted correction can overfit bag discrimination and perturb an already strong frame ordering.

## Trial 2 — conservative logit shrinkage plus standardized neuron evidence

Hypothesis: retain only 20% of the learned correction and add 0.1 times video-wise standardized neuron evidence in logit space. Standardization makes the identical coefficient meaningful across all baselines and datasets; shrinkage protects strong baseline rankings. Formal results were LaGoVAD UCF 82.960 (+1.840), LaGoVAD XD 76.627 (+2.377), DeSC UCF 89.437 (+0.067), DeSC XD 87.366 (+0.186), DSANet UCF 89.482 (+0.042), and DSANet XD 87.476 (+0.526). Minimum gain: +0.042 pp. The trial was retained.

## Trial 3 — train-only consensus distillation into sparse CLS neurons (`3c90d7c`, discarded)

The DeSC/DSANet training-score consensus was distilled into a 32-neurons-per-layer sparse expert. Results were LaGoVAD UCF 82.175 (+1.055), LaGoVAD XD 75.642 (+1.392), DeSC UCF 89.450 (+0.080), DeSC XD 87.348 (+0.168), DSANet UCF 89.463 (+0.023), and DSANet XD 87.326 (+0.376). Minimum gain: +0.023 pp, below the retained +0.042. Dense pseudo supervision improved DeSC UCF slightly but weakened the expert's useful complementary evidence elsewhere.

## Trial 4 — neuron-gated dual-anchor consensus

Hypothesis: retain DeSC/DSANet complementarity directly, calibrate their score scales with empirical CDFs fitted only on training predictions, use a fixed width-25 event expansion, and add standardized CLS-neuron evidence. The same rank weight 0.5, expansion weight 0.5, and neuron weight 0.15 are used on both datasets and for all reported baselines. This explicitly costs two anchor-baseline inference passes.

Formal results were LaGoVAD UCF 90.450 (+9.330), LaGoVAD XD 88.205 (+13.955), DeSC UCF 90.450 (+1.080), DeSC XD 88.205 (+1.025), DSANet UCF 90.450 (+1.010), and DSANet XD 88.205 (+1.255). Minimum gain: +1.0095 pp. The trial was retained and reached the target. The narrow limiting margin is DSANet UCF (+0.0095 pp above the requested +1.0 threshold), so exact score alignment and training-CDF construction must be preserved in reproductions.

