import torch
import config
from loss_schemes import getLossFunction, resetLossSchemeState
import plot

# Train Model
def trainModel(trials, params, PINN, device):
    loss_function = getLossFunction()
    resetLossSchemeState()

    PINN_params = list(PINN.parameters()) + [params.a, params.b]
    PINN_optimizer = torch.optim.Adam(PINN_params, lr=config.LEARNING_RATE_PINN)
    # - 
    if config.SPATIAL_ADAPTIVE_WEIGHTS: λ_params = [params.λ_physics, params.λ_data]
    if config.SPATIAL_ADAPTIVE_WEIGHTS: λ_optimizer = torch.optim.Adam(params=λ_params, lr=config.LEARNING_RATE_λ)

    # Training Loop 
    for epoch in range(1, config.EPOCHS_ADAM + 1):

        # 1. Zero-Grad
        PINN_optimizer.zero_grad()
        if config.SPATIAL_ADAPTIVE_WEIGHTS: λ_optimizer.zero_grad()

        # 2. Backpropagate
        ℒ = loss_function(trials=trials, params=params, PINN=PINN, epoch=epoch, device=device)
        ℒ.backward()

        # 3. Make gradient negative for λ parameters to achieve ascent, rather than descent
        if config.SPATIAL_ADAPTIVE_WEIGHTS:
            for λ in λ_params:
                if λ.grad is not None: λ.grad = -λ.grad

        # 4. Step
        PINN_optimizer.step()
        if config.SPATIAL_ADAPTIVE_WEIGHTS: λ_optimizer.step()

        # Plot
        if epoch % config.VISUALIZATION_STEP == 0: plot.plotResults(params=params, trials=trials, device=device)
        print(f'epoch: {epoch}  loss: {ℒ.item():.3f}  a: {params.a.item():.3f}   c: {params.b.item():.3f}')
