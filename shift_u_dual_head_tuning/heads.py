from __future__ import annotations

import torch
from torch import nn


def score_head_modules(adapter: nn.Module, baseline: str) -> dict[str, nn.Module]:
    if baseline == "dsanet":
        return {"classifier": adapter.base.classifier}
    if baseline == "desc":
        return {
            "sensitivity_classifier": adapter.sensitivity.classifier,
            "consistency_classifier": adapter.consistency.classifier,
        }
    if baseline == "lagovad":
        return {"bin_head": adapter.base.bin_head}
    raise ValueError(f"unknown baseline: {baseline}")


def score_head_state(adapter: nn.Module, baseline: str) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in score_head_modules(adapter, baseline).items()
    }


def load_score_head_state(
    adapter: nn.Module,
    baseline: str,
    state: dict[str, dict[str, torch.Tensor]],
) -> None:
    modules = score_head_modules(adapter, baseline)
    if set(modules) != set(state):
        raise ValueError(f"score-head checkpoint keys differ: expected={sorted(modules)}, got={sorted(state)}")
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)


def clone_score_head_parameters(adapter: nn.Module, baseline: str) -> dict[str, torch.Tensor]:
    return {
        f"{module_name}.{parameter_name}": parameter.detach().clone()
        for module_name, module in score_head_modules(adapter, baseline).items()
        for parameter_name, parameter in module.named_parameters()
    }


def relative_score_head_change(
    adapter: nn.Module,
    baseline: str,
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = []
    for module_name, module in score_head_modules(adapter, baseline).items():
        for parameter_name, parameter in module.named_parameters():
            key = f"{module_name}.{parameter_name}"
            denominator = initial[key].square().mean().clamp_min(1e-8)
            terms.append((parameter - initial[key]).square().mean() / denominator)
    if not terms:
        raise RuntimeError("score-head change requires at least one parameter")
    return torch.stack(terms).mean().sqrt()
