# Research log

## Trial 1 — sparse CLS-neuron expert and learned score correction (`9d1a066`, discarded)

The shared sparse expert selected 32 CLS hidden-state dimensions per CLIP layer and trained a small temporal expert. Its held-out video discrimination was strong (about 0.978 AUC/AP on UCF and 0.992 on XD), but the video-level correction objective did not reliably improve frame ranking. Paper-relative gains were: LaGoVAD UCF +4.857, LaGoVAD XD +3.185, DeSC UCF +0.002, DeSC XD -0.091, DSANet UCF +0.091, and DSANet XD +0.252 pp. Minimum gain: -0.091 pp. The failure indicates that unrestricted correction can overfit bag discrimination and perturb an already strong frame ordering.

## Trial 2 — conservative logit shrinkage plus standardized neuron evidence

Hypothesis: retain only 20% of the learned correction and add 0.1 times video-wise standardized neuron evidence in logit space. Standardization makes the identical coefficient meaningful across all baselines and datasets; shrinkage protects strong baseline rankings. A read-only preflight on cached curves predicted all six results above their paper baselines, with DeSC UCF as the limiting combination. Formal remote verification is pending in this trial.

## Trial 4 - dual-anchor consensus (invalidated)

This attempt combined DeSC and DSANet score streams and reached a nominal minimum paper-relative gain of +1.0095 pp. It is invalid for this project because a method evaluated for one baseline must not consume another baseline's predictions. The implementation and leaderboard rows were removed; the result is retained here only as a negative methodological record and is excluded from all future comparisons.

## New run Trial 1 - single-baseline self-calibrated event expansion

Hypothesis: strong baselines mainly miss the temporal extent of anomalous events. Calibrate each baseline only against its own training-score CDF, use a fixed width-25 temporal maximum filter, and add the existing standardized CLS-neuron evidence. Rank weight 0.5, event weight 0.5, and neuron weight 0.15 are identical for all six combinations. Formal remote verification is pending.

