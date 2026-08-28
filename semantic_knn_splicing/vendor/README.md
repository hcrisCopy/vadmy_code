# Copied upstream code

This folder contains immutable working copies required by `CodeWritingRequirement.txt`.
The training code imports these copies and never imports `baseline/` or `rely/` directly.

- `dsanet/src/`: copied from `baseline/DSANet/src`; its `LICENSE` is saved as `DSANET_LICENSE`.
- `desc_src/`: copied from `baseline/DeSC/src`; the released project did not include a separate license file.
- `lagovad_src/`: copied from `baseline/LaGoVAD-PreVAD/src`; its `LICENSE` is saved as `LAGOVAD_LICENSE`.

Do not edit these folders. Method changes belong to the parent `semantic_knn_splicing/` package.
