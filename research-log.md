# Research log

## Trial 1 — sparse CLS-neuron expert and learned score correction (`9d1a066`, discarded)

The shared sparse expert selected 32 CLS hidden-state dimensions per CLIP layer and trained a small temporal expert. Its held-out video discrimination was strong (about 0.978 AUC/AP on UCF and 0.992 on XD), but the video-level correction objective did not reliably improve frame ranking. Paper-relative gains were: LaGoVAD UCF +4.857, LaGoVAD XD +3.185, DeSC UCF +0.002, DeSC XD -0.091, DSANet UCF +0.091, and DSANet XD +0.252 pp. Minimum gain: -0.091 pp. The failure indicates that unrestricted correction can overfit bag discrimination and perturb an already strong frame ordering.

## Trial 2 — conservative logit shrinkage plus standardized neuron evidence

Hypothesis: retain only 20% of the learned correction and add 0.1 times video-wise standardized neuron evidence in logit space. Standardization makes the identical coefficient meaningful across all baselines and datasets; shrinkage protects strong baseline rankings. A read-only preflight on cached curves predicted all six results above their paper baselines, with DeSC UCF as the limiting combination. Formal remote verification is pending in this trial.

## Trial 4 - dual-anchor consensus (invalidated)

This attempt combined DeSC and DSANet score streams and reached a nominal minimum paper-relative gain of +1.0095 pp. It is invalid for this project because a method evaluated for one baseline must not consume another baseline's predictions. The implementation and leaderboard rows were removed; the result is retained here only as a negative methodological record and is excluded from all future comparisons.

## New run Trial 1 - single-baseline self-calibrated event expansion (discarded)

The method replaced the learned correction with current-baseline training-CDF calibration and an unconditional width-25 maximum filter. Formal results were LaGoVAD UCF 81.049 (-0.071), LaGoVAD XD 73.350 (-0.900), DeSC UCF 89.631 (+0.261), DeSC XD 87.763 (+0.583), DSANet UCF 89.804 (+0.364), and DSANet XD 87.065 (+0.115). Minimum gain was -0.900 pp. Unconditional peak propagation spreads normal false positives, especially for LaGoVAD, so the trial was reverted.

## New run Trial 2 - CLS-neuron-gated event expansion

Hypothesis: retain the conservative learned correction and propagate a local peak only where standardized CLS-neuron evidence independently supports anomaly. A sigmoid gate, width 25, and event weight 0.6 are fixed across all baselines and datasets. Formal results were LaGoVAD UCF 83.332 (+2.212), LaGoVAD XD 77.194 (+2.944), DeSC UCF 89.589 (+0.219), DeSC XD 87.511 (+0.331), DSANet UCF 89.719 (+0.279), and DSANet XD 87.535 (+0.585). Minimum gain was +0.219 pp, so the trial was retained.

## New run Trial 3 - full neuron-gated propagation

Hypothesis: when the CLS-neuron gate is active, use the full local event peak instead of the conservative 0.6 blend. The width-25 neighborhood and every other parameter remain unchanged across all six combinations. Formal remote verification is pending.

