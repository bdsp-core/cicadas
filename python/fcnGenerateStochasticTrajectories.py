# fcn_generateStochasticTrajectories.py
import numpy as np

def fcnGenerateStochasticTrajectories(t, params, N=100):
    """
    Generate disease trajectories with time-varying volatility.

    Args:
        t (array-like): Time vector (hours), assumed uniform spacing.
        params (array-like): [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
        N (int, optional): Number of trajectories to generate. Default 100.

    Returns:
        np.ndarray: Trajectories, shape (N, len(t))
    """
    t = np.asarray(t, dtype=float)
    params = np.asarray(params, dtype=float).reshape(-1)

    if t.size < 2:
        raise ValueError("Time vector t must have at least two points.")
    if params.size != 7:
        raise ValueError("params must have 7 elements: [gamma, H, alpha, delta, sigma_early, sigma_late, tau].")

    gamma, H, alpha, delta, sigma_early, sigma_late, tau = params
    dt = float(t[1] - t[0])
    Nt = t.size

    trajectories = np.zeros((N, Nt), dtype=float)

    for i in range(N):
        X = np.zeros(Nt, dtype=float)
        X[0] = 0.0  # initial condition

        for j in range(1, Nt):
            current_time = t[j]

            # Growth / decay dynamics
            if current_time < 30.0:
                # Logistic growth phase
                growth_term = gamma * X[j-1] * (1.0 - X[j-1] / H)
            else:
                # Decay phase
                time_since_peak = current_time - 30.0
                decay_factor = np.exp(-delta * time_since_peak)
                growth_term = -alpha * (X[j-1] - H * decay_factor * 0.2)

            # Mean reversion (prevent overshoot)
            mean_reversion = -alpha * max(0.0, X[j-1] - H)
            drift = growth_term + mean_reversion

            # Time-varying diffusion
            sigma_t = sigma_early * np.exp(-current_time / tau) + sigma_late
            diffusion = sigma_t * np.sqrt(max(0.001, X[j-1]))  # minimum variance floor

            # Euler–Maruyama step
            dW = np.sqrt(dt) * np.random.randn()
            X[j] = max(0.0, X[j-1] + drift * dt + diffusion * dW)

        trajectories[i, :] = X

    return trajectories

# Optional alias to match the MATLAB-style name (if any code refers to it)
def fcn_generateStochasticTrajectories(t, params, N=100):
    return fcnGenerateStochasticTrajectories(t, params, N)
