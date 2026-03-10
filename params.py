import torch
from torch import nn
import paths
from dataclasses import dataclass
from typing import Callable
import config
import pandas as pd
import yaml

# Parameter Data Class
@dataclass
class Params:
    u_data: torch.Tensor
    x_data: torch.Tensor
    # - 
    x_coll: torch.Tensor
    # - 
    a: nn.Parameter
    b: nn.Parameter
    f: Callable
    # - 
    mask: callable
    λ_physics: nn.Parameter
    λ_data: nn.Parameter

# Build Parameters 
def buildParams(device) -> Params:
    dtype       = torch.float32
    df          = pd.read_csv(paths.data_dir / 'data.csv')
    with open(paths.data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)
    # - 
    u_data      = torch.tensor(df["u"].values, device=device, dtype=dtype).unsqueeze(1)
    x_data      = torch.tensor(df["x"].values, device=device, dtype=dtype).unsqueeze(1)
    # -
    x_coll      = torch.rand((config.N_PTS, 1), device=device, requires_grad=True)
    # - 
    a = nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True))     # coeff.
    b = nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))     # coeff.
    f = lambda x: -100 * (x - 1) * x**10                        # forcing term
    # - 
    mask = lambda λ: λ**2  # softplus mask to ensure positivity
    λ_physics = nn.Parameter(torch.ones([config.N_PTS, 1], device=device, requires_grad=True))
    λ_data = nn.Parameter(torch.ones([u_data.shape[0], 1], device=device, requires_grad=True))

    return Params(
        u_data=u_data,
        x_data=x_data,
        # - 
        x_coll=x_coll,
        # - 
        a=a,
        b=b,
        f=f,
        # - 
        mask=mask,
        λ_physics=λ_physics,
        λ_data=λ_data,
    )
