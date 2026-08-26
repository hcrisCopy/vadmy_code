from __future__ import annotations

from torch import nn

from shift_residual_head_tuning.method import ShiftResidualInjector


class FrozenSingleResidualInjector(ShiftResidualInjector):
    """The established Shift residual, identified as the frozen-baseline control."""

    method_name = "baseline_specific_shift_single_frozen_v1"


def freeze_entire_baseline(adapter: nn.Module, injector: nn.Module) -> list[str]:
    """Freeze every baseline parameter and leave only the external injector trainable."""
    adapter.requires_grad_(False)
    injector.requires_grad_(True)
    unexpected = [
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and not name.startswith("pre_temporal_conditioner.")
    ]
    if unexpected:
        raise RuntimeError(f"frozen-baseline audit failed: {unexpected}")
    return unexpected


def frozen_baseline_train_mode(adapter: nn.Module, injector: nn.Module) -> None:
    """Keep all author modules in eval mode while optimizing the residual."""
    adapter.eval()
    injector.train()
