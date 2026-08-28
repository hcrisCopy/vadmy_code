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
    lines = command.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip().startswith("python -m universal_neuron_adapter.evaluate ")]
    if len(starts) != 1:
        raise RuntimeError(f"expected one shared evaluation command, found {len(starts)}")
    block = [lines[starts[0]]]
    index = starts[0]
    while block[-1].rstrip().endswith("\\"):
        index += 1
        if index >= len(lines):
            raise RuntimeError("unterminated shared evaluation command")
        block.append(lines[index])
    evaluation = "\n".join(block)
    if evaluation.count("--baseline-manifest") != 1:
        raise RuntimeError("the shared evaluation command must expose exactly one baseline input")
    if evaluation.count("--baseline-train-manifest") != 1:
        raise RuntimeError("normal calibration must use exactly one current-baseline training stream")
    if evaluation.count("--expert-train-manifest") != 1 or evaluation.count("--expert-manifest") != 1:
        raise RuntimeError("training and test must each use exactly one shared CLS-neuron stream")
    if evaluation.count("--expert2-manifest") != 1:
        raise RuntimeError("evaluation must use exactly one shared complementary CLS-neuron expert")
    if evaluation.count("--expert2-train-manifest") != 1:
        raise RuntimeError("calibration must use exactly one complementary training stream")
    if evaluation.count("--expert3-manifest") != 1:
        raise RuntimeError("evaluation must use exactly one shared normality CLS-neuron expert")
    if evaluation.count("--expert3-train-manifest") != 1:
        raise RuntimeError("normality calibration must use exactly one shared normality training stream")
    if evaluation.count("--student-manifest") != 1 or evaluation.count("--student-train-manifest") != 1:
        raise RuntimeError("evaluation must use one shared context-neuron test/train stream")
    if '"$source_base/baseline_train/' not in evaluation or '"$source_base/baseline_test/' not in evaluation:
        raise RuntimeError("training calibration and evaluation must use the current baseline only")
    if '--baseline "$baseline"' not in evaluation:
        raise RuntimeError("the evaluation must receive only the current loop baseline")
    if "train_diverse_expert" in command or "active64_seed3407" in command:
        raise RuntimeError("the redundant seed-and-capacity auxiliary expert is forbidden")
    if "--seed 234" not in command:
        raise RuntimeError("the complementary learned expert must share seed 234")
    print("single-baseline constraint: pass", flush=True)


if __name__ == "__main__":
    main()
