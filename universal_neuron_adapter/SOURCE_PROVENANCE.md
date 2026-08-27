# Source provenance

The implementation is original project code. It uses PyTorch, NumPy, pandas, scikit-learn, and tqdm APIs. The frozen baseline score curves, trained first-round CLS-neuron expert, and correction checkpoints are reused from autoresearch trial `9d1a066`; they were generated from the repository's read-only baseline implementations and data without copying or modifying baseline source code.

The high-confidence pseudo-label training pattern follows the weakly supervised self-training design used by the read-only `rely/MIST_VAD` reference. No code is imported from or modified inside `rely`.
