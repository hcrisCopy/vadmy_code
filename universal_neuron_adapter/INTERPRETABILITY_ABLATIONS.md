# Interpretability ablations

Run only on the remote server after `conda activate dsanet`:

```bash
bash universal_neuron_adapter/commands/run_interpretability_ablations.sh
```

Outputs are written to `../vadmy_data/universal_neuron_adapter/interpretability_ablations/<commit>/`.

The protocol is frozen before observing its test results. It contains five complementary checks adapted to the fixed-budget logic used by DNA and V-FIND:

1. selected neurons versus same-layer random, global random, and high-ranked non-selected coordinates;
2. equal-budget layer intervals 1-3, 4-6, 7-9, and 10-12;
3. 4, 8, 16, and 32 selected coordinates per layer;
4. directional class-mean activation interventions against same-layer random controls;
5. neuron discovery with 25%, 50%, and 100% of the official training subset.

All probes use the same frozen linear-readout protocol. The validation split comes from the official training set. Test labels are used only for post-hoc reporting and never select a neuron count, layer interval, threshold, or checkpoint.

## Frozen run `3f04e82`

The fixed-budget test results support neuron specificity. Values below are percentages; random controls report five-seed mean and standard deviation.

| Dataset | Selected | Same-layer random | Global random | Hard non-selected |
| --- | ---: | ---: | ---: | ---: |
| UCF AUC | 95.895 | 95.573 +/- 0.591 | 95.262 +/- 0.548 | 95.852 |
| UCF AP | 96.486 | 95.424 +/- 0.519 | 95.164 +/- 0.767 | 96.178 |
| XD AUC | 98.769 | 97.091 +/- 0.585 | 97.299 +/- 0.135 | 98.094 |
| XD AP | 99.340 | 98.179 +/- 0.468 | 98.440 +/- 0.056 | 98.886 |

Equal-budget depth localization selected layers 10-12 as the strongest group on both datasets. With 4/8/16/32 neurons per layer, test AP changed from 94.997/96.087/96.274/96.486 on UCF and 97.275/98.836/98.957/99.340 on XD.

Directional interventions were also stronger for selected neurons. Normal-to-abnormal flip rates were 5.333% versus 0.800% same-layer random on UCF and 2.667% versus 0.667% on XD. Abnormal-to-normal rates were 4.286% versus 1.429% on UCF and 3.200% versus 0.800% on XD.

Discovery-size results show gradual recovery rather than a threshold collapse:

| Dataset | Training fraction | Video AUC | Video AP | Jaccard with full selection |
| --- | ---: | ---: | ---: | ---: |
| UCF | 25% | 95.014 | 94.860 | 0.299 |
| UCF | 50% | 95.857 | 95.871 | 0.357 |
| UCF | 100% | 97.210 | 97.263 | 1.000 |
| XD | 25% | 98.454 | 99.138 | 0.243 |
| XD | 50% | 98.966 | 99.448 | 0.391 |
| XD | 100% | 99.243 | 99.608 | 1.000 |

These are post-hoc interpretability results, not additional main-method selection experiments. The moderate coordinate overlap at reduced data sizes should be described as evidence for a distributed neuron subspace, not a unique immutable neuron list.
