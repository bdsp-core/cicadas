# fcnEstimatePKPD_FixedKe_Optimized.py
# State-Space Mixed Effects Model with FIXED ke (optimized)
# Returns: theta_est, patient_params, L0_est, results

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import minimize

# ------------------------------
# Public API
# ------------------------------

def fcnEstimatePKPD_FixedKe_Optimized(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    age: np.ndarray,
    sofa: np.ndarray,
    t: np.ndarray,
    parmsL: np.ndarray,
    ke_fixed: float,
    *,
    MaxIterEM: int = 50,
    TolEM: float = 1e-4,
    Verbose: bool = True,
    RegularizationStrength: float = 2.0,
    UsePriors: bool = True,
    PriorMeans: np.ndarray | list = (3, 0.1, 0.15, 4, 0.08, 0.12),
    PriorStds: np.ndarray | list = (0.5, 0.05, 0.05, 0.5, 0.05, 0.05),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, object]]:
    """
    Python translation of:
      [theta_est, patient_params, L0_est, results] =
        fcnEstimatePKPD_FixedKe_Optimized(L_obs, A_obs, age, sofa, t, parmsL, ke_fixed, ...)

    Inputs
    ------
    L_obs, A_obs : (N, T) observed arrays
    age, sofa    : (N,) covariates
    t            : (T,) time vector
    parmsL       : (7,) natural history params [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
    ke_fixed     : float, FIXED ke value

    Keyword args mirror MATLAB inputParser fields.

    Returns
    -------
    theta_est     : (8,) = [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, sigma_C, sigma_g]   (NO ke here)
    patient_params: dict with C_indiv, g_indiv, C_pred, g_pred, ke_fixed
    L0_est        : (N, T) smoothed/estimated L0 per patient (with fixed ke)
    results       : dict (iterations, history, R2s, flags...)
    """
    # Coerce types
    L_obs = np.asarray(L_obs, dtype=float)
    A_obs = np.asarray(A_obs, dtype=float)
    age = np.asarray(age, dtype=float).reshape(-1)
    sofa = np.asarray(sofa, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    parmsL = np.asarray(parmsL, dtype=float).reshape(-1)

    N, T = L_obs.shape
    dt = float(t[1] - t[0])

    # Normalize covariates (population stats, ddof=1 like MATLAB std)
    age_mean, age_std = np.mean(age), (np.std(age, ddof=1) or 1.0)
    sofa_mean, sofa_std = np.mean(sofa), (np.std(sofa, ddof=1) or 1.0)
    age_norm = (age - age_mean) / age_std
    sofa_norm = (sofa - sofa_mean) / sofa_std

    if Verbose:
        print("\n========================================")
        print(f"FIXED ke ESTIMATION (ke = {ke_fixed:.3f})")
        print("========================================")
        print(f"Patients: {N}, Time points: {T}")
        print("Parameters to estimate: 6 (b0_C, b1_C, b2_C, b0_g, b1_g, b2_g)")
        print(f"Regularization: {RegularizationStrength:.2f}")

    # ---- Initialize parameters (6 regression + 2 variance terms) ----
    prior_means = np.asarray(PriorMeans, dtype=float).reshape(6)
    prior_stds  = np.asarray(PriorStds, dtype=float).reshape(6)

    if UsePriors:
        theta = np.concatenate([prior_means, [0.5, 0.5]]).astype(float)  # (8,)
    else:
        theta = np.array([3, 0.1, 0.1, 4, 0.1, 0.1, 0.5, 0.5], dtype=float)

    # Individual effects (jitter to break symmetry)
    rng = np.random.default_rng()
    C_indiv = theta[0] + theta[1]*age_norm + theta[2]*sofa_norm + 0.1 * rng.standard_normal(N)
    g_indiv = theta[3] + theta[4]*age_norm + theta[5]*sofa_norm + 0.1 * rng.standard_normal(N)
    C_indiv = _clip(C_indiv, 0.5, 10.0)
    g_indiv = _clip(g_indiv, 0.5, 10.0)

    # Initialize L0 using fixed ke
    L0_est = initializeL0_withFixedKe(L_obs, A_obs, C_indiv, g_indiv, ke_fixed, t)

    log_likelihood_history: list[float] = []
    theta_history: list[np.ndarray] = []

    # =========================
    # EM with FIXED ke
    # =========================
    for iter_ in range(1, MaxIterEM + 1):
        if Verbose and (iter_ % 5 == 0 or iter_ == 1):
            print(f"\n--- EM Iteration {iter_} (Fixed ke) ---")

        theta_prev = theta.copy()
        L0_prev = L0_est.copy()

        # ----- E-STEP: Update L0 with FIXED ke via EKF -----
        log_lik = 0.0
        for i in range(N):
            L0_est[i, :], _, lik_i = runEKF_fixedKe(
                L_obs[i, :], A_obs[i, :], C_indiv[i], g_indiv[i], ke_fixed, dt
            )
            log_lik += float(lik_i)

        # Add priors as pseudo-likelihood
        if UsePriors:
            prior_lik = computePriorLikelihood(theta[:6], prior_means, prior_stds)
            log_lik += RegularizationStrength * float(prior_lik)

        log_likelihood_history.append(log_lik)

        # ----- M-STEP: Update C_i, g_i then population coefs -----
        # Step 1: refine individual C,g for treated patients only (any A>0)
        has_treatment = np.nanmax(A_obs, axis=1) > 0
        for i in np.where(has_treatment)[0]:
            C_exp = theta[0] + theta[1]*age_norm[i] + theta[2]*sofa_norm[i]
            g_exp = theta[3] + theta[4]*age_norm[i] + theta[5]*sofa_norm[i]

            params_i = optimizeIndividual_fixedKe(
                L_obs[i, :], A_obs[i, :], L0_est[i, :], ke_fixed, dt,
                expected_params=np.array([C_exp, g_exp], dtype=float),
                reg_weight=RegularizationStrength,
            )

            C_indiv[i] = params_i[0]
            g_indiv[i] = params_i[1]

        valid_idx = np.isfinite(C_indiv) & np.isfinite(g_indiv) & (C_indiv > 0) & (g_indiv > 0)
        if np.sum(valid_idx) > 10:
            X = np.column_stack([np.ones(np.sum(valid_idx)), age_norm[valid_idx], sofa_norm[valid_idx]])

            # Ridge for C
            lambda_C = RegularizationStrength * 10.0
            beta_C = _ridge_closed_form(X, C_indiv[valid_idx], lambda_C, prior_means[:3])
            # Smooth update (momentum)
            momentum = 0.7
            theta[0:3] = momentum * theta[0:3] + (1 - momentum) * beta_C

            C_pred = X @ theta[0:3]
            theta[6] = float(np.sqrt(np.mean((C_indiv[valid_idx] - C_pred) ** 2)))

            # Ridge for g
            lambda_g = RegularizationStrength * 10.0
            beta_g = _ridge_closed_form(X, g_indiv[valid_idx], lambda_g, prior_means[3:6])
            theta[3:6] = momentum * theta[3:6] + (1 - momentum) * beta_g

            g_pred = X @ theta[3:6]
            theta[7] = float(np.sqrt(np.mean((g_indiv[valid_idx] - g_pred) ** 2)))

        # Shrink individual effects toward population predictions
        C_pop = theta[0] + theta[1]*age_norm + theta[2]*sofa_norm
        g_pop = theta[3] + theta[4]*age_norm + theta[5]*sofa_norm
        shrinkage = 0.2
        C_indiv = (1 - shrinkage) * C_indiv + shrinkage * C_pop
        g_indiv = (1 - shrinkage) * g_indiv + shrinkage * g_pop
        C_indiv = _clip(C_indiv, 0.5, 10.0)
        g_indiv = _clip(g_indiv, 0.5, 10.0)

        theta_history.append(theta.copy())

        # ----- Convergence -----
        if iter_ > 1:
            param_change = _rel_change(theta, theta_prev)
            L0_change = _rel_change(L0_est, L0_prev)
            if Verbose and (iter_ % 5 == 0 or iter_ == 2):
                print(f"  Parameter change: {param_change:.6f}")
                print(f"  L0 change:        {L0_change:.6f}")
                print(f"  Log-likelihood:   {log_lik:.2f}")

            if (param_change < TolEM) and (L0_change < TolEM):
                if Verbose:
                    print(f"\nConverged after {iter_} iterations")
                break

    # Compile outputs
    theta_est = theta.copy()  # 8 parameters (NO ke)

    patient_params = {
        "C_indiv": C_indiv,
        "g_indiv": g_indiv,
        "C_pred": theta[0] + theta[1]*age_norm + theta[2]*sofa_norm,
        "g_pred": theta[3] + theta[4]*age_norm + theta[5]*sofa_norm,
        "ke_fixed": float(ke_fixed),
    }

    # R²
    results: Dict[str, object] = {}
    valid_C = np.isfinite(C_indiv)
    valid_g = np.isfinite(g_indiv)

    if np.sum(valid_C) > 10:
        num = np.sum((C_indiv[valid_C] - patient_params["C_pred"][valid_C]) ** 2)
        den = np.sum((C_indiv[valid_C] - np.mean(C_indiv[valid_C])) ** 2) + 1e-12
        results["R2_C"] = 1.0 - num / den
    else:
        results["R2_C"] = np.nan

    if np.sum(valid_g) > 10:
        num = np.sum((g_indiv[valid_g] - patient_params["g_pred"][valid_g]) ** 2)
        den = np.sum((g_indiv[valid_g] - np.mean(g_indiv[valid_g])) ** 2) + 1e-12
        results["R2_g"] = 1.0 - num / den
    else:
        results["R2_g"] = np.nan

    results["iterations"] = iter_
    results["log_likelihood_history"] = np.asarray(log_likelihood_history, dtype=float)
    results["theta_history"] = np.asarray(theta_history, dtype=float).T if theta_history else np.empty((8, 0))
    results["converged"] = (iter_ < MaxIterEM)
    results["ke_was_fixed"] = True
    results["ke_value"] = float(ke_fixed)

    if Verbose:
        print("\n========================================")
        print(f"FINAL ESTIMATES (with FIXED ke = {ke_fixed:.3f}):")
        print(f"C = {theta[0]:.3f} + {theta[1]:.3f}*age + {theta[2]:.3f}*sofa (σ={theta[6]:.3f})")
        print(f"g = {theta[3]:.3f} + {theta[4]:.3f}*age + {theta[5]:.3f}*sofa (σ={theta[7]:.3f})")
        print(f"R² for C: {results['R2_C']:.3f}, R² for g: {results['R2_g']:.3f}")
        print("========================================")

    return theta_est, patient_params, L0_est, results


# ------------------------------
# Helpers (translate MATLAB subfuncs)
# ------------------------------

def initializeL0_withFixedKe(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    C: np.ndarray,
    g: np.ndarray,
    ke: float,
    t: np.ndarray,
) -> np.ndarray:
    """Initialize L0 using inverse treatment effect with FIXED ke (per MATLAB)."""
    N, T = L_obs.shape
    L0_init = L_obs.copy()
    X = np.zeros(T, dtype=float)

    for i in range(N):
        X[:] = 0.0
        for j in range(1, T):
            X[j] = ke * X[j - 1] + A_obs[i, j]

        for j in range(T):
            if X[j] > 0 and np.isfinite(L_obs[i, j]):
                sX = 1.0 - 1.0 / ((C[i] / X[j]) ** g[i] + 1.0)
                if 0.1 < sX < 0.9:
                    L0_init[i, j] = L_obs[i, j] / sX
                else:
                    L0_init[i, j] = L_obs[i, j]

        # Smooth and bound
        L0_init[i, :] = _smooth_1d(L0_init[i, :], window=5)
        L0_init[i, :] = _clip(L0_init[i, :], 0.0, 5.0)

    return L0_init


def runEKF_fixedKe(
    L_i: np.ndarray,
    A_i: np.ndarray,
    C: float,
    g: float,
    ke: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    EKF with FIXED ke.
    State: L0 (scalar); Observation: L = sX * L0, where sX depends on X (ke-filtered A).
    """
    T = L_i.size
    L0_filt = np.zeros(T, dtype=float)
    P_filt = np.zeros(T, dtype=float)
    L0_filt[0] = float(L_i[0])
    P_filt[0] = 0.1

    Q = 0.01
    R = 0.05

    # Build X with fixed ke
    X = np.zeros(T, dtype=float)
    for t_ in range(1, T):
        X[t_] = ke * X[t_ - 1] + A_i[t_]

    log_lik = 0.0

    for t_ in range(1, T):
        # Predict
        L0_pred = L0_filt[t_ - 1]
        P_pred = P_filt[t_ - 1] + Q

        if np.isfinite(L_i[t_]):
            if X[t_ - 1] > 0:
                sX = 1.0 - 1.0 / ((C / X[t_ - 1]) ** g + 1.0)
                H = sX
            else:
                sX = 1.0
                H = 1.0

            y_pred = L0_pred * sX
            innovation = L_i[t_] - y_pred

            S = H * P_pred * H + R
            K = (P_pred * H) / S

            L0_filt[t_] = L0_pred + K * innovation
            P_filt[t_] = (1.0 - K * H) * P_pred

            if S > 0:
                log_lik -= 0.5 * (np.log(2.0 * np.pi * S) + (innovation ** 2) / S)
        else:
            L0_filt[t_] = L0_pred
            P_filt[t_] = P_pred

    # Smooth and clip
    L0_smooth = _smooth_1d(L0_filt, window=5)
    L0_smooth = _clip(L0_smooth, 0.0, np.inf)
    return L0_smooth, P_filt, float(log_lik)


def optimizeIndividual_fixedKe(
    L_i: np.ndarray,
    A_i: np.ndarray,
    L0_i: np.ndarray,
    ke: float,
    dt: float,
    expected_params: np.ndarray,
    reg_weight: float,
) -> np.ndarray:
    """Optimize C and g with FIXED ke (SLSQP bounds to mirror fmincon)."""
    def obj(p):
        return individualNLL_fixedKe(p, L_i, A_i, L0_i, ke, expected_params, reg_weight)

    bounds = [(0.5, 10.0), (0.5, 10.0)]
    res = minimize(obj, x0=np.asarray(expected_params, float), method="SLSQP", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-9, "disp": False})
    return res.x.astype(float)


def individualNLL_fixedKe(
    params: np.ndarray,
    L_i: np.ndarray,
    A_i: np.ndarray,
    L0_i: np.ndarray,
    ke: float,
    expected_params: np.ndarray,
    reg_weight: float,
) -> float:
    """Per-patient NLL with FIXED ke (matches MATLAB structure)."""
    C, g = float(params[0]), float(params[1])
    T = L_i.size

    X = np.zeros(T, dtype=float)
    for t_ in range(1, T):
        X[t_] = ke * X[t_ - 1] + A_i[t_]

    nll = 0.0
    n_obs = 0

    for t_ in range(1, T):
        if np.isfinite(L_i[t_]) and (L0_i[t_] > 0) and (X[t_ - 1] >= 0):
            if X[t_ - 1] > 0:
                sX = 1.0 - 1.0 / ((C / X[t_ - 1]) ** g + 1.0)
                L_pred = L0_i[t_] * sX
            else:
                L_pred = L0_i[t_]

            var_L = 0.01 + 0.05 * abs(L_pred)
            if (var_L > 0) and (L_pred >= 0):
                nll += 0.5 * np.log(2.0 * np.pi * var_L) + 0.5 * ((L_i[t_] - L_pred) ** 2) / var_L
                n_obs += 1

    if n_obs > 0:
        nll /= n_obs

    # L2 regularization around expected params
    nll += reg_weight * float(np.sum((np.asarray(params) - np.asarray(expected_params)) ** 2))
    return float(nll)


def computePriorLikelihood(theta: np.ndarray, prior_means: np.ndarray, prior_stds: np.ndarray) -> float:
    """Gaussian prior likelihood (log), same shape as MATLAB helper."""
    z = (np.asarray(theta) - np.asarray(prior_means)) / np.asarray(prior_stds)
    return float(-0.5 * np.sum(z ** 2))


# ------------------------------
# Small utilities (smooth, ridge, etc.)
# ------------------------------

def _ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float, prior_means_3: np.ndarray) -> np.ndarray:
    # (X'X + lam*I)^{-1} (X'y + lam*prior)
    XT = X.T
    A = XT @ X + lam * np.eye(3)
    b = XT @ y + lam * np.asarray(prior_means_3, float)
    beta = np.linalg.solve(A, b)
    return beta

def _smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple centered moving average (fallback if short)."""
    x = np.asarray(x, float)
    w = int(max(1, window))
    if x.size < w or w == 1:
        return x.copy()
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(xpad, kernel, mode="valid")
    return y

def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(arr, float), lo, hi)

def _rel_change(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a.ravel())
    nb = np.linalg.norm(b.ravel())
    denom = na + 1e-10
    return float(np.linalg.norm((a - b).ravel()) / denom)
