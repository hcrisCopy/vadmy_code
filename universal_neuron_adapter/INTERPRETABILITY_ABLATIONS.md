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
