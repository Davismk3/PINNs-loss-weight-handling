# Loss Handling (PINN)

Adaptive Loss Strategies for Physics-Informed Neural Networks

This repository implements and compares multiple loss balancing and adaptive weighting strategies for training Physics-Informed Neural Networks (PINNs). These methods address a key challenge in PINN training: competing loss terms arising from physics residuals, boundary conditions, and observational data.

The codebase is designed as a modular experimentation framework for studying how different weighting schemes affect convergence, stability, and solution accuracy in PDE-constrained learning problems.

## Problem

The project targets:

$$
-\nabla \cdot (a \nabla u(x)) + b \cdot u(x) = f(x)
$$

where `a` and `b` are learned during training.

## Results

Results show that gradient-normalized global self-adaptive weighting significantly outperforms fixed or unweighted schemes, while spatially adaptive weighting provides a small additional improvement. 

## Repository Layout

```text
.
├── main.py              # entrypoint
├── training.py          # training loop + optimizers
├── architecture.py      # Fourier features + PINN + trial function
├── loss.py              # physics/data residual definitions
├── loss_schemes.py      # baseline + gradient-normalized loss modes
├── params.py            # data loading + trainable parameters
├── config.py            # experiment configuration
├── paths.py             # project path helpers
├── plot.py              # live plotting during training
└── data/
    ├── data.csv
    └── parameters.yaml
```

## Requirements

- Python 3.10+
- PyTorch
- pandas
- matplotlib
- PyYAML

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pandas matplotlib pyyaml
```

## Run

```bash
python main.py
```

`main.py` selects device in this order when `USE_GPU=True`:

1. Apple Metal (`mps`)
2. CUDA
3. CPU fallback

## Configuration

Edit [`config.py`](./config.py) to control training:

- `EPOCHS_ADAM`: number of optimization steps
- `LEARNING_RATE_PINN`: learning rate for network + PDE coefficients
- `SPATIAL_ADAPTIVE_WEIGHTS`: enables trainable per-point lambda weights
- `GLOBAL_ADAPTIVE_WEIGHTS`: switches to gradient-normalization loss scheme
- `VISUALIZATION_STEP`: plotting frequency
- `A`, `B`: boundary values used in the trial function

## Data Format

`data/data.csv` must include:

- `x`: spatial coordinate
- `u`: observed solution value

Current sample:

```csv
x,u
0.9,1.0
0.4,0.1
```

## Notes

- Coefficients `a` and `b` are optimized jointly with network weights.
- The forcing term is currently defined in `params.py` as:
  `f(x) = -100 * (x - 1) * x**10`
- Live plots are generated during training via `matplotlib`.

## Suggested GitHub Cleanup Before Push

Add a `.gitignore` for local artifacts such as:

- `__pycache__/`
- `.DS_Store`
- `.venv/`

