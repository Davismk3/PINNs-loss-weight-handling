import torch
import torch.nn as nn
import torch.nn.init as init
from dataclasses import dataclass
from typing import Callable
import config

# Fourier Input Transformation Layer (Tancik et al.) 
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, neurons, scale):
        super().__init__()  # inherit
        self.B = nn.Parameter(torch.randn(in_dim, neurons) * scale)

    def forward(self, x): 
        proj = x @ self.B  # cant do 'x @= self.B' because it breaks gradients?
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# Simple PINN Architecture
def buildSimplePINN(neurons):
    PINN = nn.Sequential(

        # 1 Fourier Input Layer
        FourierFeatures(in_dim=1, neurons=neurons, scale=config.SCALE),
        
        # 5 Hidden Layers
        nn.Linear(in_features=neurons * 2, out_features=neurons),
        config.ACTIVATION,
        # -
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # -
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,
        # - 
        nn.Linear(in_features=neurons, out_features=neurons),
        config.ACTIVATION,

        # 1 Linear Output Layer
        nn.Linear(in_features=neurons, out_features=2)
    )
    return PINN

# Trial Function Data Class
@dataclass 
class Trials:
    u_trial: Callable

# Build Trial Functions
def buildTrials(params, device, PINN) -> Trials:
    x_data = params.x_data.squeeze(1)
    u_data = params.u_data.squeeze(1)

    default_A = getattr(config, "A", None)
    if default_A is None:
        idx_A = torch.argmin(torch.abs(x_data - 0.0))
        A = float(u_data[idx_A].detach().cpu())
    else:
        A = float(default_A)

    default_B = getattr(config, "B", None)
    if default_B is None:
        idx_B = torch.argmin(torch.abs(x_data - 1.0))
        B = float(u_data[idx_B].detach().cpu())
    else:
        B = float(default_B)

    u_trial = lambda x: A * (1 - x) + B * x + PINN(torch.cat([x], dim=1))[:, 0:1] * (1 - x) * x
    
    return Trials(u_trial=u_trial)
    
