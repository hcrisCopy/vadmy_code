from __future__ import annotations

from pathlib import Path


FORBIDDEN_TOKENS = (
    "consensus_evaluate",
    "--desc-train",
    "--dsanet-train",
    "--desc-test",
    "--dsanet-test",
    "--lagovad-test",
)


def main() -> None:
    package = Path(__file__).resolve().parent
    command = (package / "commands" / "run_all.sh").read_text(encoding="utf-8")
    lowered = command.lower()
    found = [token for token in FORBIDDEN_TOKENS if token in lowered]
    if found:
        raise RuntimeError(f"cross-baseline inputs are forbidden: {found}")
    if command.count("--baseline-manifest") != 1:
        raise RuntimeError("the shared evaluation command must expose exactly one baseline input")
    if command.count("--baseline-train-manifest") != 1:
        raise RuntimeError("normal calibration must use exactly one current-baseline training stream")
    if command.count("--expert-train-manifest") != 1 or command.count("--expert-manifest") != 1:
        raise RuntimeError("training and test must each use exactly one shared CLS-neuron stream")
    if command.count("--selected-manifest") != 2:
        raise RuntimeError("semantic training and evaluation must use the shared selected CLS-neuron cache")
    if '"$source_base/baseline_train/' not in command or '"$source_base/baseline_test/' not in command:
        raise RuntimeError("training calibration and evaluation must use the current baseline only")
    if '--baseline "$baseline"' not in command:
        raise RuntimeError("the evaluation must receive only the current loop baseline")
    print("single-baseline constraint: pass", flush=True)


if __name__ == "__main__":
    main()
