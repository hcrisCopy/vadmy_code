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
        raise RuntimeError("calibration must use exactly one training stream from the current baseline")
    if '"$source_base/baseline_train/' not in command or '"$source_base/baseline_test/' not in command:
        raise RuntimeError("training and test inputs must both come from the current baseline")
    if '--baseline "$baseline"' not in command:
        raise RuntimeError("the evaluation must receive only the current loop baseline")
    print("single-baseline constraint: pass", flush=True)


if __name__ == "__main__":
    main()
