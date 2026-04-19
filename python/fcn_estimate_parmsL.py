# fcn_estimate_parmsL.py
# Estimate [gamma, H, alpha, delta, sigma_early, sigma_late, tau] from untreated L trajectories

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import minimize
from scipy.stats import ks_2samp

# Uses your existing generator
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories


def fcn_estimate_parmsL(
    trajectories: np.ndarray,
    t: np.ndarray,
    true_params: Optional[np.ndarray] = None,  # optional, mirrors MATLAB comment
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Python translation of fcn_estimate_parmsL.m

    Parameters
    ----------
    trajectories : (N, Nt) array
        Observed disease trajectories (untreated).
    t : (Nt,) array
        Time vector (hours).
    true_params : (7,) array-like, optional
        If provided, diagnostic param errors are included.

    Returns
    -------
    theta_est : (7,) array
        [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
    diagnostics : dict
        Keys: initial_estimates, p_peaks, p_onset, mse_mean, var_error,
              (optional) param_errors
    """
    trajectories = np.asarray(trajectories, dtype=float)
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    N, Nt = trajectories.shape

    # ----------------------------
    # Step 1: heuristic initials
    # ----------------------------
    early_growth_rates = np.zeros(N)
    for i in range(N):
        X = trajectories[i, :]
        peak_val = np.nanmax(X)
        if not np.isfinite(peak_val) or peak_val <= 0:
            early_growth_rates[i] = 0.0
            continue
        idx_5 = _first_idx(X > 0.05 * peak_val)
        idx_50 = _first_idx(X > 0.5 * peak_val)
        if idx_5 is not None and idx_50 is not None and idx_50 > idx_5:
            t_seg = t[idx_5:idx_50 + 1]
            X_seg = X[idx_5:idx_50 + 1]
            X_norm = X_seg / peak_val
            valid = (X_norm > 0) & (X_norm < 0.8) & np.isfinite(X_norm)
            if np.count_nonzero(valid) > 3:
                y = np.log(X_norm[valid] / (1.0 - X_norm[valid]))
                p = np.polyfit(t_seg[valid], y, 1)  # slope ~ growth rate
                early_growth_rates[i] = p[0]
        # else remains 0.0

    pos_growth = early_growth_rates[early_growth_rates > 0]
    growth_rate_init = float(np.median(pos_growth)) if pos_growth.size > 0 else 0.25  # guard

    decay_rates = np.zeros(N)
    for i in range(N):
        X = trajectories[i, :]
        peak_idx = int(np.nanargmax(X))
        peak_val = X[peak_idx]
        if not np.isfinite(peak_val) or peak_val <= 0:
            decay_rates[i] = 0.0
            continue
        if peak_idx < Nt - 20:
            t_decay = t[peak_idx:] - t[peak_idx]
            X_decay = X[peak_idx:]
            valid = (X_decay > 0.1 * peak_val) & np.isfinite(X_decay) & (X_decay > 0)
            if np.count_nonzero(valid) > 10:
                p = np.polyfit(t_decay[valid], np.log(X_decay[valid]), 1)
                decay_rates[i] = -p[0]  # positive delta
        # else remains 0.0

    pos_decay = decay_rates[decay_rates > 0]
    decay_rate_init = float(np.median(pos_decay)) if pos_decay.size > 0 else 0.05  # guard

    theta_init = np.array([
        growth_rate_init,  # gamma
        1.0,               # H
        0.15,              # alpha
        decay_rate_init,   # delta
        0.15,              # sigma_early
        0.03,              # sigma_late
        40.0               # tau
    ], dtype=float)

    # ----------------------------
    # Step 2: MLE via bound minimization
    # ----------------------------
    # Bounds (lb, ub) from MATLAB
    lb = np.array([0.05, 0.5, 0.01, 0.001, 0.01, 0.001, 10.0], dtype=float)
    ub = np.array([1.00, 2.0, 0.50, 0.100, 0.30, 0.100, 100.0], dtype=float)
    bounds = list(zip(lb, ub))

    def nll(theta: np.ndarray) -> float:
        return _compute_nll(theta, trajectories, t, dt)

    # Use L-BFGS-B (box constraints). SLSQP is another option.
    res = minimize(
        nll, theta_init, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=100, ftol=1e-9, maxls=50)
    )
    theta_est = res.x.astype(float)

    # ----------------------------
    # Step 3: simulate for diagnostics
    # ----------------------------
    N_sim = 100
    sim = fcnGenerateStochasticTrajectories(t, theta_est, N_sim)

    # ----------------------------
    # Goodness-of-fit diagnostics
    # ----------------------------
    obs_peaks = np.nanmax(trajectories, axis=1)
    sim_peaks = np.nanmax(sim, axis=1)

    obs_onset = _compute_onset_times(trajectories, t)
    sim_onset = _compute_onset_times(sim, t)

    # KS tests (two-sample)
    # Filter finite values to avoid issues
    obs_peaks_f = obs_peaks[np.isfinite(obs_peaks)]
    sim_peaks_f = sim_peaks[np.isfinite(sim_peaks)]
    if obs_peaks_f.size > 1 and sim_peaks_f.size > 1:
        _, p_peaks = ks_2samp(obs_peaks_f, sim_peaks_f, alternative="two-sided", mode="auto")
    else:
        p_peaks = np.nan

    if obs_onset.size > 1 and sim_onset.size > 1:
        _, p_onset = ks_2samp(obs_onset, sim_onset, alternative="two-sided", mode="auto")
    else:
        p_onset = np.nan

    # Mean / percentiles / variance comparisons
    obs_mean = np.nanmean(trajectories, axis=0)
    sim_mean = np.nanmean(sim, axis=0)

    mse_diff = (obs_mean - sim_mean) ** 2
    mse_mean = float(np.nanmean(mse_diff)) if np.isfinite(mse_diff).any() else np.nan

    obs_var = np.nanvar(trajectories, axis=0)
    sim_var = np.nanvar(sim, axis=0)
    denom = obs_var + 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        var_diff = np.abs(obs_var - sim_var) / denom
    var_error = float(np.nanmean(var_diff)) if np.isfinite(var_diff).any() else np.nan

    diagnostics: Dict[str, Any] = dict(
        initial_estimates=theta_init,
        p_peaks=p_peaks,
        p_onset=p_onset,
        mse_mean=mse_mean,
        var_error=var_error,
        success=bool(res.success),
        message=res.message,
        nfev=int(res.nfev),
        njev=int(getattr(res, "njev", 0)),
        fun=float(res.fun),
    )

    if true_params is not None:
        true_params = np.asarray(true_params, dtype=float).reshape(-1)
        if true_params.size == 7 and np.all(np.isfinite(true_params)):
            diagnostics["param_errors"] = np.abs(theta_est - true_params)

    return theta_est, diagnostics


# ---------------------------------------------------------------------
# Helpers (mirroring MATLAB inner functions)
# ---------------------------------------------------------------------
def _compute_nll(theta: np.ndarray, trajectories: np.ndarray, t: np.ndarray, dt: float) -> float:
    gamma, H, alpha, delta, sigma_early, sigma_late, tau = theta
    N, Nt = trajectories.shape
    nll = 0.0

    for i in range(N):
        X = trajectories[i, :]
        for j in range(1, Nt):
            xprev = X[j - 1]
            xcurr = X[j]
            if not np.isfinite(xcurr) or not np.isfinite(xprev):
                continue
            if xprev <= 0:
                continue

            current_time = t[j]

            # Drift
            if current_time < 30.0:
                growth_term = gamma * xprev * (1.0 - xprev / H)
            else:
                time_since_peak = current_time - 30.0
                decay_factor = np.exp(-delta * time_since_peak)
                growth_term = -alpha * (xprev - H * decay_factor * 0.2)

            mean_reversion = -alpha * max(0.0, xprev - H)
            drift = growth_term + mean_reversion

            # Diffusion (time-varying)
            sigma_t = sigma_early * np.exp(-current_time / tau) + sigma_late
            diffusion_sq = sigma_t ** 2 * max(0.001, xprev)

            mu_t = xprev + drift * dt
            var_t = diffusion_sq * dt

            if var_t <= 0.0:
                continue

            # Gaussian negative log-likelihood (up to constant)
            # nll += 0.5*log(2πσ^2) + (x - μ)^2/(2σ^2)
            nll += 0.5 * np.log(2.0 * np.pi * var_t) + ((xcurr - mu_t) ** 2) / (2.0 * var_t)

    # Quadratic regularization to discourage extremes (same spirit as MATLAB)
    nll += 0.1 * float(np.sum(theta ** 2))
    return nll


def _compute_onset_times(trajectories: np.ndarray, t: np.ndarray) -> np.ndarray:
    N, _ = trajectories.shape
    onset = []
    for i in range(N):
        X = trajectories[i, :]
        peak_val = np.nanmax(X)
        if not np.isfinite(peak_val) or peak_val <= 0:
            continue
        thr = 0.1 * peak_val
        idx = _first_idx(X > thr)
        if idx is not None:
            onset.append(t[idx])
    return np.asarray(onset, dtype=float)


def _first_idx(mask: np.ndarray) -> Optional[int]:
    """Return first index where mask is True, else None."""
    mask = np.asarray(mask, dtype=bool)
    idxs = np.flatnonzero(mask)
    return int(idxs[0]) if idxs.size > 0 else None
