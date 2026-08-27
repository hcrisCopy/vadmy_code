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
    if command.count("--baseline-manifest") != 2:
        raise RuntimeError("training and evaluation must each expose exactly one current-baseline input")
    if command.count('"$source_base/baseline_train/') != 1:
        raise RuntimeError("training must read only the current baseline training stream")
    if command.count('"$source_base/baseline_test/') != 1:
        raise RuntimeError("evaluation must read only the current baseline test stream")
    if '--baseline "$baseline"' not in command:
        raise RuntimeError("the evaluation must receive only the current loop baseline")
    print("single-baseline constraint: pass", flush=True)


if __name__ == "__main__":
    main()
