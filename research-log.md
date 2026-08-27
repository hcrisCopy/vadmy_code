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

## New run Trials 26-28 - rejected normality uses

Direct normality residual reduced the minimum to +0.629 pp. Category normality gating reduced it to +0.609 pp. Replacing the retained video classifier with a normality-aware classifier improved DSANet UCF to +0.755 but reduced DeSC UCF to +0.650, so all three trials were reverted.

## New run Trial 29 - consensus normality video suppression

Hypothesis: the normality-aware video classifier contains useful DSANet normal-video evidence but should not replace the retained classifier. Preserve the retained one-sided shift and add 0.25 times the normality-aware negative decision only when both classifiers agree the current video is normal. Formal results were LaGoVAD UCF 86.726 (+5.606), LaGoVAD XD 79.272 (+5.022), DeSC UCF 90.030 (+0.660), DeSC XD 87.836 (+0.656), DSANet UCF 90.110 (+0.670), and DSANet XD 87.905 (+0.955). Minimum gain was +0.656 pp, narrowly above Trial 25, so the trial was retained.

## New run Trial 30 - rebalanced neuron-gated event support

Hypothesis: after adding the strong normality gate, width 51 propagates peaks slightly too far on XD, while width 31 loses UCF support. A read-only universal scan found width 41 gave a +0.664 pp minimum gain versus +0.646 at width 31 and +0.656 for the retained width 51. Formal results were LaGoVAD UCF 86.720 (+5.600), LaGoVAD XD 79.270 (+5.020), DeSC UCF 90.034 (+0.664), DeSC XD 87.846 (+0.666), DSANet UCF 90.134 (+0.694), and DSANet XD 87.893 (+0.943). Minimum gain was +0.664 pp, so width 41 was retained.

Post-Trial-30 diagnostics rejected learned three-expert snippet fusion (+0.620 minimum), normal-suppression weights 1.0/2.0 (+0.661/+0.647), persistence widths 11/19 (+0.642/+0.628), a pure-neuron video normality classifier (+0.664), normality power transforms 0.5/1.5 (+0.661/+0.663), positive video shifts (at most +0.500), top-gated positive shifts (at most +0.642), and larger direct-neuron weights 0.5/0.6 (+0.630/+0.575). These variants either relearned noisy MIL localization or improved UCF while corrupting XD cross-video ordering.

## New run Trial 31 - high-high baseline-neuron agreement residual

Hypothesis: a larger unconditional neuron residual helps UCF but hurts XD because it also changes baseline-neuron conflicts. Add a 0.1 logit residual only where the current single baseline and the first CLS-neuron expert are both above their within-video means, using their minimum standardized evidence. No low-low or conflicting snippet is changed by this new branch. Formal results were LaGoVAD UCF 86.834 (+5.714), LaGoVAD XD 79.258 (+5.008), DeSC UCF 90.059 (+0.689), DeSC XD 87.845 (+0.665), DSANet UCF 90.139 (+0.699), and DSANet XD 87.906 (+0.956). Minimum gain was +0.665 pp, so the trial was retained.

## New run Trial 32 - training-calibrated persistence scale

Hypothesis: a fixed snippet window ignores the temporal scale visible in the shared neuron experts. For each dataset, apply the same baseline-independent rule: measure the longest positive three-expert consensus run in every training video, multiply its 75th percentile by 0.35, round to the nearest odd integer, and clip to 7–21. The rule yields 15 on UCF and 11 on XD without branching on dataset identity or baseline. Formal results were LaGoVAD UCF 86.834 (+5.714), LaGoVAD XD 79.252 (+5.002), DeSC UCF 90.059 (+0.689), DeSC XD 87.868 (+0.688), DSANet UCF 90.139 (+0.699), and DSANet XD 87.941 (+0.991). Minimum gain was +0.688 pp, so the adaptive scale was retained.

## New run Trials 33-34 - full adaptive persistence projection

Hypothesis: once the persistence width is calibrated from training neuron dynamics, mixing 25% of the unfiltered curve retains isolated false positives. A universal scan found persistence weights 0.5 and 1.0 produced minimum gains of +0.621 and +0.700 pp, respectively. Trial 33 was infrastructure-invalid: GitHub returned HTTP 503 during the remote pull, and the remote shell continued with the retained 0.75 configuration. Trial 34 formally produced LaGoVAD UCF 86.832 (+5.712), LaGoVAD XD 79.266 (+5.016), DeSC UCF 90.070 (+0.700), DeSC XD 87.892 (+0.712), DSANet UCF 90.151 (+0.711), and DSANet XD 87.900 (+0.950). Minimum gain was +0.700 pp, so full adaptive persistence was retained.

## New run Trial 35 - balanced high-high agreement strength

Hypothesis: full persistence removes the XD sensitivity that previously limited the high-high residual. A universal scan of agreement weights 0.15 and 0.20 yielded minimum gains of +0.711405 and +0.711358 pp. Formal results at 0.15 were LaGoVAD UCF 86.890 (+5.770), LaGoVAD XD 79.256 (+5.006), DeSC UCF 90.081 (+0.711), DeSC XD 87.892 (+0.712), DSANet UCF 90.153 (+0.713), and DSANet XD 87.905 (+0.955). Minimum gain was +0.711 pp, so weight 0.15 was retained.

Post-Trial-35 diagnostics found Gaussian sigma 0.5 marginally improved the minimum to +0.7118, whereas sigma 1.5 fell to +0.606. Event widths 31 and 51 reached only +0.696 and +0.690. These results confirm short edge smoothing and width 41, but neither changes the main bottleneck.

## New run Trial 36 - directional one-sided normality neurons

Hypothesis: absolute normal z-deviation treats both tails as anomalous even when a neuron has a consistent abnormal direction. For every CLS coordinate, rank above-normal and below-normal top-k effects separately, retain the stronger direction, select 32 coordinates per layer, and aggregate only positive directed deviations. The expert remains baseline-independent and uses the same algorithm on both datasets. Formal results were LaGoVAD UCF 86.849 (+5.729), LaGoVAD XD 79.369 (+5.119), DeSC UCF 90.082 (+0.712), DeSC XD 88.040 (+0.860), DSANet UCF 90.163 (+0.723), and DSANet XD 87.978 (+1.028). Minimum gain was +0.712 pp, so the directional expert was retained.

## New run Trial 37 - correction-head removal and stronger direct neuron evidence

Hypothesis: the original learned correction head is redundant after directional normality and full persistence. Scanning correction weights 0.0/0.1/0.3 gave minima +0.7126/+0.7125/+0.7121 pp. With correction removed, direct-neuron weights 0.5/0.6/0.8 gave UCF minima +0.7195/+0.7222/+0.7129. Formal results at weight 0.6 were LaGoVAD UCF 86.490 (+5.370), LaGoVAD XD 78.114 (+3.864), DeSC UCF 90.155 (+0.785), DeSC XD 88.072 (+0.892), DSANet UCF 90.162 (+0.722), and DSANet XD 87.986 (+1.036). Minimum gain was +0.722 pp, so correction removal and direct weight 0.6 were retained.

## New run Trial 38 - stronger directional normality consensus

Hypothesis: the one-sided expert can carry more gate weight than the former absolute-deviation expert because the learned direction removes the irrelevant tail. A UCF scan over weights 2.5/3.0/4.0/5.0 peaked at 3.0, where DSANet gained +0.728 pp; larger values declined. Formal results were LaGoVAD UCF 86.452 (+5.332), LaGoVAD XD 77.559 (+3.309), DeSC UCF 90.141 (+0.771), DeSC XD 87.968 (+0.788), DSANet UCF 90.168 (+0.728), and DSANet XD 87.829 (+0.879). Minimum gain was +0.728 pp, so weight 3.0 was retained.

## New run Trial 39 - reduced one-sided normal-video suppression

Hypothesis: directional normality already rejects many false events, so the retained video-level normal shift of 1.5 over-suppresses DSANet UCF. A universal scan gave minimum gains +0.741 at suppression weight 1.0 and +0.607 at 0.5. Formal results at weight 1.0 were LaGoVAD UCF 86.241 (+5.121), LaGoVAD XD 77.301 (+3.051), DeSC UCF 90.135 (+0.765), DeSC XD 87.962 (+0.782), DSANet UCF 90.181 (+0.741), and DSANet XD 87.770 (+0.820). Minimum gain was +0.741 pp, so suppression weight 1.0 was retained.

Post-Trial-39 diagnostics rejected event weights 0.8/1.2 (+0.671/+0.105 minimum), normality-video auxiliary coefficients 0/0.5 (UCF minima +0.732/+0.740), direct signed-normality residuals (best UCF +0.735), and direct diverse-expert residuals (best UCF +0.736).

## New run Trial 40 - triple high-confidence neuron agreement

Hypothesis: a residual should be added only when the current baseline, first MIL neuron expert, and directional normality expert all exceed their within-video means. Use the minimum of the three positive standardized evidences, leaving every conflict and low-evidence snippet unchanged. A universal scan found weight 0.8 balanced UCF at +0.7748 minimum and XD at +0.7787; weight 1.2 overfit UCF and reduced XD to +0.702. Formal remote verification is pending.
