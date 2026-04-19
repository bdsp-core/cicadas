import numpy as np
from typing import Sequence, Optional


def fcn_generateStochasticTrajectories(t: Sequence[float], params: Sequence[float], N: Optional[int] = None) -> np.ndarray:
    """
    Direct translation of MATLAB:
        trajectories = fcn_generateStochasticTrajectories(t, params, N)

    Inputs:
        t      : time vector (hours), length Nt (assumed uniform step)
        params : [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
                 gamma       - Growth rate
                 H           - Peak height
                 alpha       - Mean reversion rate
                 delta       - Decay rate
                 sigma_early - Early volatility
                 sigma_late  - Late volatility
                 tau         - Volatility decay time constant
        N      : number of trajectories (default 100, matching MATLAB)

    Output:
        trajectories : (N, Nt) array of disease trajectories
    """
    # Default N like: if nargin < 3, N = 100
    if N is None:
        N = 100

    # --- Extract parameters ---
    params = np.asarray(params, dtype=float)
    gamma = params[0]
    H = params[1]
    alpha = params[2]
    delta = params[3]
    sigma_early = params[4]
    sigma_late = params[5]
    tau = params[6]

    # --- Setup ---
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("t must have at least 2 points to compute a time step.")
    dt = t[1] - t[0]  # assume uniform
    Nt = t.size
    trajectories = np.zeros((N, Nt), dtype=float)

    # --- Generate each trajectory ---
    for i in range(N):
        X = np.zeros(Nt, dtype=float)
        X[0] = 0.0  # initial condition

        for j in range(1, Nt):
            current_time = t[j]

            # Growth dynamics
            if current_time < 30:
                # Logistic growth phase
                growth_term = gamma * X[j - 1] * (1.0 - X[j - 1] / H)
            else:
                # Decay phase
                time_since_peak = current_time - 30.0
                decay_factor = np.exp(-delta * time_since_peak)
                growth_term = -alpha * (X[j - 1] - H * decay_factor * 0.2)

            # Mean reversion term
            mean_reversion = -alpha * max(0.0, X[j - 1] - H)
            drift = growth_term + mean_reversion

            # Time-varying diffusion
            sigma_t = sigma_early * np.exp(-current_time / tau) + sigma_late
            diffusion = sigma_t * np.sqrt(max(0.001, X[j - 1]))  # minimum variance floor

            # Euler–Maruyama update
            dW = np.sqrt(dt) * np.random.randn()
            X[j] = max(0.0, X[j - 1] + drift * dt + diffusion * dW)

        trajectories[i, :] = X

    return trajectories


# CamelCase alias to match your driver call exactly
def fcnGenerateStochasticTrajectories(t: Sequence[float], params: Sequence[float], N: Optional[int] = None) -> np.ndarray:
    return fcn_generateStochasticTrajectories(t, params, N)
