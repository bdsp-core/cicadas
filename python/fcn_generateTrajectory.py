# fcn_generateTrajectory.py

from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Tuple, Union

ParamLike = Union[Dict[str, float], Iterable[float], np.ndarray]

def fcn_generateTrajectory(
    params: ParamLike,
    N: int = 100,
    T: float = 170.0,
    dt: float = 0.1,
    return_t: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate disease trajectories with time-varying volatility (Euler–Maruyama).

    Parameters
    ----------
    params : dict or sequence of 7 floats
        If dict, must contain keys:
          {'growth_rate','peak_height','alpha','decay_rate',
           'sigma_early','sigma_late','sigma_transition'}.
        If sequence/array, order is:
          [growth_rate, peak_height, alpha, decay_rate,
           sigma_early, sigma_late, sigma_transition].
    N : int, default 100
        Number of trajectories to simulate.
    T : float, default 170.0
        Total simulation time (hours).
    dt : float, default 0.1
        Time step (hours).
    return_t : bool, default False
        If True, also return the time vector.

    Returns
    -------
    trajectories : (N, Nt) ndarray
        Each row is one trajectory.
    t : (Nt,) ndarray, optional
        Returned only if return_t=True.

    Notes
    -----
    Dynamics mirror the MATLAB version:
      - Logistic growth phase until t < 30.
      - Post-peak decay with exponential factor.
      - Mean reversion to limit overshoot.
      - Time-varying diffusion:
            sigma_t = sigma_early * exp(-t / sigma_transition) + sigma_late
        with diffusion term sigma_t * sqrt(max(0.001, X_{t-1})).
    Brownian increment dW ~ N(0, dt). Uses NumPy's global RNG
    (set np.random.seed(...) outside for reproducibility).
    """
    # ---- Extract parameters ----
    if isinstance(params, dict):
        required = [
            "growth_rate", "peak_height", "alpha", "decay_rate",
            "sigma_early", "sigma_late", "sigma_transition",
        ]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing parameter keys: {missing}")
        growth_rate   = float(params["growth_rate"])
        peak_height   = float(params["peak_height"])
        alpha         = float(params["alpha"])
        decay_rate    = float(params["decay_rate"])
        sigma_early   = float(params["sigma_early"])
        sigma_late    = float(params["sigma_late"])
        sigma_trans   = float(params["sigma_transition"])
    else:
        arr = np.asarray(list(params), dtype=float).ravel()
        if arr.size != 7:
            raise ValueError("Parameter vector must have 7 elements.")
        growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_trans = arr

    # ---- Time vector (inclusive of T, like 0:dt:T in MATLAB) ----
    Nt = int(np.floor(T / dt + 1e-12)) + 1
    t = np.linspace(0.0, Nt - 1, Nt) * dt  # avoids FP drift, guarantees length

    # ---- Allocate output ----
    trajectories = np.zeros((N, Nt), dtype=float)

    # ---- Simulate trajectories ----
    sqrt_dt = np.sqrt(dt)
    for i in range(N):
        X = np.zeros(Nt, dtype=float)
        # X[0] already 0
        for j in range(1, Nt):
            current_time = t[j]

            # Growth / decay dynamics
            if current_time < 30.0:
                # Logistic growth
                growth_term = growth_rate * X[j - 1] * (1.0 - X[j - 1] / peak_height)
            else:
                # Exponential decay around 30h “peak”
                time_since_peak = current_time - 30.0
                decay_factor = np.exp(-decay_rate * time_since_peak)
                growth_term = -alpha * (X[j - 1] - peak_height * decay_factor * 0.2)

            # Mean reversion to cap overshoot
            mean_reversion = -alpha * max(0.0, X[j - 1] - peak_height)

            drift = growth_term + mean_reversion

            # Time-varying diffusion
            sigma_t = sigma_early * np.exp(-(current_time / sigma_trans)) + sigma_late
            diffusion = sigma_t * np.sqrt(max(0.001, X[j - 1]))

            # Euler–Maruyama step
            dW = sqrt_dt * np.random.randn()
            X[j] = max(0.0, X[j - 1] + drift * dt + diffusion * dW)

        trajectories[i, :] = X

    return (trajectories, t) if return_t else trajectories
