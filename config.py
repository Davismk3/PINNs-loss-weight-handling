import torch.nn as nn

USE_GPU = True
SCALE = 1
ACTIVATION = nn.Tanh()
N_PTS = 100  # random collocation pts (training)
NEURONS = 64  # hidden‑layer width
EPOCHS_ADAM = 15000  # iterations
BUFFER = 1  # perhaps have this learnable?
SCHEDULER_PATIENCE = 100  # Epochs to wait for loss improvement before reducing LR
SCHEDULER_FACTOR = 0.90  # Factor to multiply LR by when reducing (e.g., 0.5 halves it)
SCHEDULER_MIN_LR = 1e-6  # Minimum LR to stop reducing below this
LEARNING_RATE_PINN = 1e-3 
LEARNING_RATE_λ = 1e-1
A = 0.0
B = 1.0
VISUALIZATION_STEP = 10

GRAD_NORM_EPOCH_INTERVAL = 10
GRAD_NORM_EMA = 0.9
LOSS_SCALE_PHYSICS = 1.0
LOSS_SCALE_DATA = 1.0
ADAPTIVE_WEIGHT_MIN = 1e-4
ADAPTIVE_WEIGHT_MAX = 1e4

GLOBAL_ADAPTIVE_WEIGHTS = False
SPATIAL_ADAPTIVE_WEIGHTS = True 
