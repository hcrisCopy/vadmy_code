# Source provenance

The neuron adapter and analysis code are original project code using PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, and tqdm APIs. `baseline_adapters.py`, `baseline_losses.py`, and `desc_inference.py` preserve the already-audited project-side wrappers of the three authors' released inference protocols; they are local copies so the final package does not depend on a discarded experiment package. The protected `baseline/` sources are never modified.

Frozen baseline score curves, the first sparse CLS-neuron expert, and correction checkpoints are reused from autoresearch trial `9d1a066`. Hidden-state extraction follows the repository's existing DSANet CLIP ViT-B/16 loader and saves only the 12 block-level CLS tokens, not patch tokens or optical flow.

The complementary expert reuses the audited sparse-MIL implementation but applies a fixed diagonal coordinate projector derived from the frozen primary selection. The information-fusion module solves the three-dimensional non-negative regularized Fisher quadratic exactly by active-set enumeration. These are neuron-space operations, not additional baseline score anchors. The temporal stack retains the dilation pattern in the protected `rely/LAP/model.py` reference; reference code is never imported or modified.
