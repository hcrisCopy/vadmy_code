from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vin_vad.event_chain import EventChain


def evaluate_pattern(chain: EventChain, emissions: torch.Tensor) -> dict[str, object]:
    mask = torch.ones_like(emissions, dtype=torch.bool)
    log_p1 = chain.video_log_probs(emissions, mask)[1]
    posterior = chain.snippet_marginals(emissions, mask)[0]
    return {
        "video_prob": float(log_p1.exp()[0]),
        "posterior_peak": float(posterior.max()),
        "posterior_mass": float(posterior.sum()),
        "posterior": [float(value) for value in posterior],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-emission event-chain behavior audit.")
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()
    output = Path(args.out_path)
    if "vadmy_data" not in output.resolve().parts:
        raise ValueError("out-path must be inside the sibling vadmy_data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    chains = {variant: EventChain(variant).eval() for variant in ("e1", "e2", "e3")}
    isolated = torch.zeros(1, 64)
    isolated[0, 32] = 4.0
    continuous = torch.zeros(1, 64)
    continuous[0, 30:35] = 0.8
    result = {
        "patterns": {
            variant: {
                "isolated_peak": evaluate_pattern(chain, isolated),
                "continuous_medium": evaluate_pattern(chain, continuous),
            }
            for variant, chain in chains.items()
        },
        "zero_emission_length_prior": {
            variant: {
                str(length): evaluate_pattern(chain, torch.zeros(1, length))["video_prob"]
                for length in (16, 64, 256, 1024)
            }
            for variant, chain in chains.items()
        },
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
