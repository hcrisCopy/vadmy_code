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

Hypothesis: when the CLS-neuron gate is active, use the full local event peak instead of the conservative 0.6 blend. The width-25 neighborhood and every other parameter remain unchanged across all six combinations. Formal results were LaGoVAD UCF 83.586 (+2.466), LaGoVAD XD 77.409 (+3.159), DeSC UCF 89.703 (+0.333), DeSC XD 87.525 (+0.345), DSANet UCF 89.821 (+0.381), and DSANet XD 87.440 (+0.490). Minimum gain was +0.333 pp, so the trial was retained.

## New run Trial 4 - current-baseline-guided dense neuron correction (discarded)

High-confidence pseudo labels from each current baseline trained a new correction head with CLS-neuron weighting. Results were LaGoVAD UCF 82.694 (+1.574), LaGoVAD XD 76.722 (+2.472), DeSC UCF 89.669 (+0.299), DeSC XD 87.381 (+0.201), DSANet UCF 89.817 (+0.377), and DSANet XD 87.465 (+0.515). Minimum gain was +0.201 pp, below the retained +0.333. The head mostly copied its teacher and weakened DeSC/XD, so it was reverted.

## New run Trial 5 - one-sided current-baseline normal suppression

Hypothesis: fit the same five-feature video classifier on each current baseline's own training curves and use it only to suppress high-confidence normal test videos. Suspected abnormal videos receive no video-level boost, preventing their normal snippets from being raised. Formal results were LaGoVAD UCF 83.208 (+2.088), LaGoVAD XD 77.271 (+3.021), DeSC UCF 89.754 (+0.384), DeSC XD 87.561 (+0.381), DSANet UCF 89.821 (+0.381), and DSANet XD 87.487 (+0.537). Minimum gain was +0.381 pp, so the trial was retained.

## New run Trial 6 - persistent neuron-gated events

Hypothesis: isolated corrected peaks are likely false alarms, whereas true events persist across neighboring snippets. Blend the retained score with a width-15 median filter at weight 0.75 after neuron-gated expansion and one-sided normal suppression. Formal results were LaGoVAD UCF 83.219 (+2.099), LaGoVAD XD 77.359 (+3.109), DeSC UCF 89.781 (+0.411), DeSC XD 87.708 (+0.528), DSANet UCF 89.854 (+0.414), and DSANet XD 87.477 (+0.527). Minimum gain was +0.411 pp, so the trial was retained.

## New run Trial 7 - joint baseline-neuron normal suppression

Hypothesis: combine distribution, top-k, temporal-change, and agreement statistics from the current baseline and the CLS-neuron expert in the same linear video classifier. Use its output only for one-sided normal suppression; event propagation and persistence are unchanged. Formal results were LaGoVAD UCF 85.812 (+4.692), LaGoVAD XD 79.107 (+4.857), DeSC UCF 89.812 (+0.442), DeSC XD 87.715 (+0.535), DSANet UCF 89.959 (+0.519), and DSANet XD 87.569 (+0.619). Minimum gain was +0.442 pp, so the trial was retained.

## New run Trial 8 - current-baseline/CLS agreement temporal head (discarded)

A compact temporal head used agreement pseudo-labels between the current baseline and the CLS-neuron expert. Formal results were LaGoVAD UCF 86.760 (+5.640), LaGoVAD XD 75.862 (+1.612), DeSC UCF 90.211 (+0.841), DeSC XD 86.942 (-0.238), DSANet UCF 89.703 (+0.263), and DSANet XD 87.050 (+0.100). Minimum gain was -0.238 pp, so the trial was reverted. A read-only global weight and agreement-gate scan peaked below the retained result, showing that the pseudo-label target does not transfer reliably to strong baselines.

## New run Trial 9 - video-label MIL neuron residual (discarded)

A zero-initialized residual head consumed the current baseline, shared expert, and fixed 384 CLS coordinates, with video-label MIL, normal dense loss, ranking, smoothness, and a baseline anchor. Formal results were LaGoVAD UCF 87.083 (+5.963), LaGoVAD XD 77.216 (+2.966), DeSC UCF 89.828 (+0.458), DeSC XD 86.582 (-0.598), DSANet UCF 89.326 (-0.114), and DSANet XD 85.983 (-0.967). Minimum gain was -0.967 pp, so the trial was reverted. Low bag-level validation loss again failed to imply good frame localization.

## New run Trial 10 - stronger one-sided normal suppression

Hypothesis: the retained joint current-baseline/CLS video classifier is reliable only as a normal-video suppressor. Increase its universal logit shift from 1.0 to 1.5 while leaving suspected abnormal videos and all temporal operations unchanged. Formal results were LaGoVAD UCF 85.949 (+4.829), LaGoVAD XD 79.118 (+4.868), DeSC UCF 89.813 (+0.443), DeSC XD 87.713 (+0.533), DSANet UCF 89.923 (+0.483), and DSANet XD 87.573 (+0.623). Minimum gain was +0.443 pp, so the trial was retained.

## New run Trial 11 - layer-normalized neuron agreement head (discarded)

The selected coordinates were normalized within each CLIP layer and snippet before training the same current-baseline/CLS agreement head. Formal results were LaGoVAD UCF 86.887 (+5.767), LaGoVAD XD 78.758 (+4.508), DeSC UCF 90.029 (+0.659), DeSC XD 87.747 (+0.567), DSANet UCF 89.544 (+0.104), and DSANet XD 87.815 (+0.865). Minimum gain was +0.104 pp, so the trial was reverted. Layer normalization did not resolve the cross-view pseudo-label localization error.

## New run Trial 12 - wider neuron-gated event support

Hypothesis: anomaly events persist beyond the retained width-25 neighborhood. Formal results were LaGoVAD UCF 85.995 (+4.875), LaGoVAD XD 79.227 (+4.977), DeSC UCF 89.837 (+0.467), DeSC XD 87.678 (+0.498), DSANet UCF 89.936 (+0.496), and DSANet XD 87.642 (+0.692). Minimum gain was +0.467 pp, so width 51 was retained.

## New run Trial 13 - semantic multi-label neuron probes (discarded)

One baseline-independent linear probe per anomaly category was trained over the fixed 384 CLS neurons. Formal results were LaGoVAD UCF 86.103 (+4.983), LaGoVAD XD 79.486 (+5.236), DeSC UCF 89.843 (+0.473), DeSC XD 87.772 (+0.592), DSANet UCF 89.504 (+0.064), and DSANet XD 87.768 (+0.818). Minimum gain was +0.064 pp, so the trial was reverted. Finer video semantics still did not make top-k MIL a reliable frame localizer.

## New run Trial 14 - one-snippet temporal alignment

Hypothesis: corrected responses lag the frame annotations by one 16-frame snippet. Formal results were LaGoVAD UCF 85.997 (+4.877), LaGoVAD XD 79.299 (+5.049), DeSC UCF 89.862 (+0.492), DeSC XD 87.685 (+0.505), DSANet UCF 89.950 (+0.510), and DSANet XD 87.648 (+0.698). Minimum gain was +0.492 pp, so the one-snippet within-video advance was retained.

## New run Trial 15 - diverse sparse-neuron expert

Hypothesis: one 32-neurons-per-layer expert has insufficient localization diversity. Train a second baseline-independent expert with 64 active CLS dimensions per layer and the same MIL objective, then average the video-standardized evidence of both experts only in the event-propagation gate. The expert is trained once per dataset and shared by all three baselines; no second baseline score is used. Formal results were LaGoVAD UCF 85.978 (+4.858), LaGoVAD XD 79.431 (+5.181), DeSC UCF 89.873 (+0.503), DeSC XD 87.735 (+0.555), DSANet UCF 89.954 (+0.514), and DSANet XD 87.715 (+0.765). Minimum gain was +0.503 pp, so the trial was retained.

## New run Trial 16 - stronger direct neuron correction

Hypothesis: the diverse expert improved all six metrics, but the direct neuron residual remains underweighted relative to event propagation. Increase the universal direct neuron weight from 0.1 to 0.2 for every dataset and baseline. Formal results were LaGoVAD UCF 86.208 (+5.088), LaGoVAD XD 79.539 (+5.289), DeSC UCF 89.924 (+0.554), DeSC XD 87.766 (+0.586), DSANet UCF 89.963 (+0.523), and DSANet XD 87.804 (+0.854). Minimum gain was +0.523 pp, so the trial was retained.

## New run Trial 17 - neuron residual saturation probe

Hypothesis: the direct CLS-neuron residual is still below its useful saturation point. Increase its universal weight from 0.2 to 0.4 while retaining all other operations. Formal results were LaGoVAD UCF 86.528 (+5.408), LaGoVAD XD 79.561 (+5.311), DeSC UCF 89.998 (+0.628), DeSC XD 87.765 (+0.585), DSANet UCF 89.968 (+0.528), and DSANet XD 87.900 (+0.950). Minimum gain was +0.528 pp, so the trial was retained.

## New run Trials 18-19 - dual-expert direct evidence (discarded)

Using the mean of both sparse experts as direct evidence reduced the minimum gain to +0.516 pp. Requiring pointwise minimum agreement reduced it to +0.504 pp. Both trials were reverted; expert diversity helps the event gate but adds noise to direct snippet ranking.

## New run Trial 20 - short-scale Gaussian temporal denoising

Hypothesis: after width-15 persistent filtering, residual one-snippet jitter still creates false-positive ordering errors for strong baselines. Apply a universal Gaussian filter with sigma 1 snippet to the final within-video curve before the retained one-snippet alignment. Formal results were LaGoVAD UCF 86.535 (+5.415), LaGoVAD XD 79.586 (+5.336), DeSC UCF 90.003 (+0.633), DeSC XD 87.797 (+0.617), DSANet UCF 89.970 (+0.530), and DSANet XD 87.944 (+0.994). Minimum gain was +0.530 pp, so the trial was retained.

## New run Trial 21 - fractional temporal alignment (discarded)

A universal 1.5-snippet interpolated advance reduced the minimum gain to +0.523 pp, so the trial was reverted. The retained one-snippet alignment remains preferable.

## New run Trial 22 - baseline-independent normality neuron expert

Hypothesis: both existing experts learn from abnormal-video top-k MIL and therefore share localization errors. Fit normal neuron moments using only normal training videos, select 32 CLS dimensions per layer whose top-k deviation separates abnormal training videos, and add their snippet-level deviation at weight 0.5 only to the universal event gate. Formal results were LaGoVAD UCF 86.585 (+5.465), LaGoVAD XD 79.553 (+5.303), DeSC UCF 90.031 (+0.661), DeSC XD 87.840 (+0.660), DSANet UCF 90.039 (+0.599), and DSANet XD 87.961 (+1.011). Minimum gain was +0.599 pp, so the trial was retained.

## New run Trial 23 - stronger normality event gating

Hypothesis: the baseline-independent normality expert improved the UCF strong baselines without destabilizing XD. Increase only its universal event-gate coefficient from 0.5 to 0.75 while preserving neuron selection and all other operations. Formal results were LaGoVAD UCF 86.605 (+5.485), LaGoVAD XD 79.509 (+5.259), DeSC UCF 90.038 (+0.668), DeSC XD 87.848 (+0.668), DSANet UCF 90.064 (+0.624), and DSANet XD 87.957 (+1.007). Minimum gain was +0.624 pp, so the trial was retained.

## New run Trial 24 - saturated normality gate strength

Hypothesis: the normality evidence remains useful above weight 0.75. A shared scan over 1.0, 1.25, 1.5, and 2.0 found the best minimum gain at 1.5; 2.0 already degraded XD DeSC. Formal results were LaGoVAD UCF 86.617 (+5.497), LaGoVAD XD 79.268 (+5.018), DeSC UCF 90.026 (+0.656), DeSC XD 87.839 (+0.659), DSANet UCF 90.093 (+0.653), and DSANet XD 87.907 (+0.957). Minimum gain was +0.653 pp, so the trial was retained.

Normality capacity diagnostics found top-16 (+0.645) and top-64 (+0.626) below the retained top-32 setting. Pure normality smoothing also reduced the minimum because it harmed XD DeSC despite improving UCF DSANet.

## New run Trial 25 - raw/persistent normality blend

Hypothesis: raw normality evidence preserves short XD events, while sigma-1 evidence suppresses isolated UCF false positives. Blend 75% raw and 25% smoothed evidence before video standardization, using one fixed ratio for all datasets and baselines. Formal results were LaGoVAD UCF 86.617 (+5.497), LaGoVAD XD 79.260 (+5.010), DeSC UCF 90.027 (+0.657), DeSC XD 87.836 (+0.656), DSANet UCF 90.100 (+0.660), and DSANet XD 87.901 (+0.951). Minimum gain was +0.656 pp, so the trial was retained.

## New run Trials 26-27 - direct and category normality (discarded)

Adding normality directly to corrected logits reduced the minimum gain to +0.629 pp. Category-conditional normality gating improved UCF DSANet but harmed XD and reduced the minimum to +0.609 pp; an agreement-gated category scan also stayed below the retained result. Both branches were reverted.

## New run Trial 28 - normality-aware one-sided video suppression

Hypothesis: the retained one-sided normal-video classifier lacks the independently useful global normality statistics. Add normality curve distribution, correlation, and disagreement features to the same logistic classifier trained from current-baseline training videos, and continue using its decision only to suppress likely normal test videos. No abnormal-video boost or cross-baseline input is introduced. Formal remote verification is pending.
