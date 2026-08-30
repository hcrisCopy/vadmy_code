from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def initialize_experiment(output: Path, name: str, seed: int) -> dict[str, object]:
    if not VALID_NAME.fullmatch(name):
        raise ValueError(
            "experiment name must use 1-80 letters, digits, dots, dashes or underscores"
        )
    payload: dict[str, object] = {
        "experiment_name": name,
        "seed": seed,
        "git_revision": git_revision(),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"{output} belongs to another code version or seed; "
                "choose a new EXPERIMENT_NAME"
            )
    else:
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify a named formal experiment directory."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    payload = initialize_experiment(Path(args.out_dir), args.name, args.seed)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
