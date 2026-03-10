import torch
from dataclasses import dataclass
from typing import Callable

import config
from loss import uDataLoss, uPhysicsLoss

EPS = 1e-12


@dataclass
class _GradNormState:
    lambda_physics_old: float = 1.0
    lambda_data_old: float = 1.0


_grad_norm_state = _GradNormState()


def resetLossSchemeState():
    global _grad_norm_state
    _grad_norm_state = _GradNormState()


def _safeRatio(num, den):
    return num / (den + EPS)


def _clampAdaptiveWeight(weight):
    min_weight = float(getattr(config, "ADAPTIVE_WEIGHT_MIN", 0.0))
    max_weight = float(getattr(config, "ADAPTIVE_WEIGHT_MAX", float("inf")))
    return torch.clamp(weight, min=min_weight, max=max_weight)


def _maxGradMagnitude(loss_value, PINN, params):
    param_list = list(PINN.parameters()) + [params.a, params.b]
    grads = torch.autograd.grad(
        loss_value,
        param_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    max_abs = 0.0
    for grad in grads:
        if grad is not None:
            grad_max = torch.max(torch.abs(grad))
            if torch.isfinite(grad_max):
                max_abs = max(max_abs, float(grad_max.detach().cpu()))
    return torch.tensor(max_abs, device=loss_value.device, dtype=loss_value.dtype).clamp_min(EPS)


def _meanAbsGrad(loss_value, PINN, params):
    param_list = list(PINN.parameters()) + [params.a, params.b]
    grads = torch.autograd.grad(
        loss_value,
        param_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_abs = 0.0
    count = 0
    for grad in grads:
        if grad is not None:
            grad_abs = torch.abs(grad)
            finite_mask = torch.isfinite(grad_abs)
            if finite_mask.any():
                total_abs += float(torch.sum(grad_abs[finite_mask]).detach().cpu())
                count += int(finite_mask.sum().item())
    if count == 0:
        return torch.tensor(EPS, device=loss_value.device, dtype=loss_value.dtype)
    return torch.tensor(total_abs / count, device=loss_value.device, dtype=loss_value.dtype).clamp_min(EPS)


def _rawLossTerms(params, trials):
    u_physics_term = uPhysicsLoss(params, trials)
    u_data_term = uDataLoss(params, trials)
    loss_physics_raw = 0.5 * torch.mean(params.mask(params.λ_physics) * u_physics_term**2)
    loss_data_raw = 0.5 * torch.mean(params.mask(params.λ_data) * u_data_term**2)
    return loss_physics_raw, loss_data_raw


def baselineLoss(trials, params, PINN=None, epoch=None, device=None):
    loss_physics_raw, loss_data_raw = _rawLossTerms(params=params, trials=trials)
    return torch.sqrt(loss_physics_raw.clamp_min(EPS)) + torch.sqrt(loss_data_raw.clamp_min(EPS))


def lossGradientNormalizationMax(trials, params, PINN, epoch, device):
    global _grad_norm_state
    loss_physics_raw, loss_data_raw = _rawLossTerms(params=params, trials=trials)

    grad_interval = max(1, int(getattr(config, "GRAD_NORM_EPOCH_INTERVAL", 1)))
    if epoch % grad_interval == 0:
        grad_phys = _maxGradMagnitude(loss_physics_raw, PINN, params)
        grad_data = _meanAbsGrad(loss_data_raw, PINN, params)
        grad_reference = grad_phys

        lambda_physics = _clampAdaptiveWeight(_safeRatio(grad_reference, grad_phys))
        lambda_data = _clampAdaptiveWeight(_safeRatio(grad_reference, grad_data))

        ema = float(getattr(config, "GRAD_NORM_EMA", 0.9))
        _grad_norm_state.lambda_physics_old = ema * _grad_norm_state.lambda_physics_old + (1.0 - ema) * float(
            lambda_physics.detach().cpu()
        )
        _grad_norm_state.lambda_data_old = ema * _grad_norm_state.lambda_data_old + (1.0 - ema) * float(
            lambda_data.detach().cpu()
        )

    scale_physics = float(getattr(config, "LOSS_SCALE_PHYSICS", 1.0))
    scale_data = float(getattr(config, "LOSS_SCALE_DATA", 1.0))
    loss_physics = loss_physics_raw * scale_physics * _grad_norm_state.lambda_physics_old
    loss_data = loss_data_raw * scale_data * _grad_norm_state.lambda_data_old

    return torch.sqrt(loss_physics.clamp_min(EPS)) + torch.sqrt(loss_data.clamp_min(EPS))


LOSS_SCHEMES: dict[str, Callable] = {
    "baselineLoss": baselineLoss,
    "lossGradientNormalizationMax": lossGradientNormalizationMax,
}


def getLossFunction():
    if config.GLOBAL_ADAPTIVE_WEIGHTS: return lossGradientNormalizationMax
    return baselineLoss
