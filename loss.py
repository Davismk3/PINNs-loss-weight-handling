import torch

def uPhysicsLoss(params, trials):
    x = params.x_coll
    u = trials.u_trial(x)
    # - 
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    diffusion_term = - torch.autograd.grad((params.a * u_x), x, torch.ones_like(u), create_graph=True)[0]
    # - 
    return diffusion_term + params.b * u - params.f(x)

def uDataLoss(params, trials):    
    point_u = params.u_data
    point_x = params.x_data

    return trials.u_trial(point_x) - point_u

def totalLoss(params, trials): 
    u_physics_term = uPhysicsLoss(params, trials)
    u_data_term = uDataLoss(params, trials)

    # Normalize everything 
    # u_physics_term = u_physics_term / (u_physics_term.detach() + BUFFER)
    # u_data_term = u_data_term / (u_data_term.detach() + BUFFER)
    # NOTE this did NOT work at all, so it is commented out

    # Individual losses, in the style of McClenny & Braga-Neto, all become scalars
    ℒ_physics = 0.5 * torch.mean(params.mask(params.λ_physics) * u_physics_term**2)
    ℒ_data = 0.5 * torch.mean(params.mask(params.λ_data) * u_data_term**2)

    # loss function, in the style of McClenny & Braga-Neto
    # ℒ = ℒ_physics + ℒ_data
    ℒ = torch.sqrt(ℒ_physics) + torch.sqrt(ℒ_data)  # sqrt helps converge faster, from Urbán et al.

    return ℒ
