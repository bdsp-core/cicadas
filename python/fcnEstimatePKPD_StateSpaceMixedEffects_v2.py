# fcnEstimatePKPD_StateSpaceMixedEffects_v2.py
# Enhanced State-Space Mixed Effects (unfixed ke)
# Returns: theta_est, patient_params, L0_est, results

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def fcnEstimatePKPD_StateSpaceMixedEffects_v2(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    age: np.ndarray,
    sofa: np.ndarray,
    t: np.ndarray,
    parmsL: np.ndarray,
    *,
    MaxIterEM: int = 50,
    TolEM: float = 1e-4,
    Verbose: bool = True,
    InitMethod: str = "informed",
    RegularizationStrength: float = 1.0,
    UsePriors: bool = True,
    PriorMeans: np.ndarray | list = (3, 0.1, 0.15, 4, 0.08, 0.12, 0.5),  # last is ke
    PriorStds: np.ndarray | list = (1, 0.05, 0.05, 1, 0.05, 0.05, 0.2),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, object]]:
    """
    Python translation of:
      [theta_est, patient_params, L0_est, results] =
        fcnEstimatePKPD_StateSpaceMixedEffects_v2(L_obs, A_obs, age, sofa, t, parmsL, ...)

    Inputs
    ------
    L_obs, A_obs : (N, T)
    age, sofa    : (N,)
    t            : (T,)
    parmsL       : (7,) [gamma, H, alpha, delta, sigma_early, sigma_late, tau]

    Keyword args mirror MATLAB inputParser options.

    Returns
    -------
    theta_est      : (9,) = [b0_C,b1_C,b2_C,b0_g,b1_g,b2_g,ke, sigma_C, sigma_g]
    patient_params : dict with C_indiv, g_indiv, C_pred, g_pred
    L0_est         : (N, T) smoothed/estimated L0
    results        : dict (iterations, histories, convergence flags, R2s, data_quality)
    """
    # Coerce types/shapes
    L_obs = np.asarray(L_obs, float)
    A_obs = np.asarray(A_obs, float)
    age = np.asarray(age, float).reshape(-1)
    sofa = np.asarray(sofa, float).reshape(-1)
    t = np.asarray(t, float).reshape(-1)
    parmsL = np.asarray(parmsL, float).reshape(-1)

    N, T = L_obs.shape
    dt = float(t[1] - t[0])

    # Normalize covariates (MATLAB-style ddof=1)
    age_mean, age_std = np.mean(age), (np.std(age, ddof=1) or 1.0)
    sofa_mean, sofa_std = np.mean(sofa), (np.std(sofa, ddof=1) or 1.0)
    age_norm = (age - age_mean) / age_std
    sofa_norm = (sofa - sofa_mean) / sofa_std

    if Verbose:
        print("\n========================================")
        print("ENHANCED STATE-SPACE MIXED EFFECTS")
        print("========================================")
        print(f"Patients: {N}, Time points: {T}")
        print(f"Regularization strength: {RegularizationStrength:.2f}")
        print(f"Using priors: {bool(UsePriors)}")

    # ----- Initialization -----
    prior_means = np.asarray(PriorMeans, float).reshape(7)   # includes ke
    prior_stds  = np.asarray(PriorStds, float).reshape(7)

    if UsePriors:
        # [b0_C,b1_C,b2_C,b0_g,b1_g,b2_g,ke, sigma_C, sigma_g]
        theta = np.concatenate([prior_means, [0.5, 0.5]]).astype(float)
        if Verbose:
            print("Initialized with prior means")
    else:
        theta = np.array([3, 0.1, 0.1, 4, 0.1, 0.1, 0.5, 0.5, 0.5], float)

    # Individual params near population
    rng = np.random.default_rng()
    C_indiv = theta[0] + theta[1]*age_norm + theta[2]*sofa_norm + 0.1 * rng.standard_normal(N)
    g_indiv = theta[3] + theta[4]*age_norm + theta[5]*sofa_norm + 0.1 * rng.standard_normal(N)
    C_indiv = _clip(C_indiv, 0.5, 10.0)
    g_indiv = _clip(g_indiv, 0.5, 10.0)

    # Initial L0 (improved)
    L0_est = initializeL0_improved(L_obs, A_obs, C_indiv, g_indiv, theta[6], parmsL, t)

    # Histories
    log_likelihood_history: list[float] = []
    theta_history: list[np.ndarray] = []

    # Adaptive regularization weights per patient
    data_quality = assessDataQuality(L_obs, A_obs)
    reg_weights = computeRegularizationWeights(data_quality, RegularizationStrength)

    # ----- EM loop -----
    momentum = 0.7  # used in multiple updates
    for iter_ in range(1, MaxIterEM + 1):
        if Verbose and (iter_ % 5 == 0 or iter_ == 1):
            print(f"\n--- EM Iteration {iter_} ---")

        theta_prev = theta.copy()
        L0_prev = L0_est.copy()

        # E-STEP: Enhanced EKF for each patient
        log_lik = 0.0
        for i in range(N):
            Q_proc = 0.01 * (1.0 + 2.0*(1.0 - data_quality[i]))
            R_meas = 0.05 * (1.0 + 3.0*(1.0 - data_quality[i]))
            L0_est[i, :], _, lik_i = runEKF_improved(
                L_obs[i, :], A_obs[i, :], C_indiv[i], g_indiv[i],
                theta[6], parmsL, dt, Q_proc, R_meas
            )
            log_lik += float(lik_i)

        if UsePriors:
            prior_lik = computePriorLikelihood(theta[:7], prior_means, prior_stds)
            log_lik += RegularizationStrength * float(prior_lik)

        log_likelihood_history.append(log_lik)

        # M-STEP: 1) Update individual C,g (regularized toward population)
        has_treatment = np.nanmax(A_obs, axis=1) > 0
        for i in np.where(has_treatment)[0]:
            C_exp = theta[0] + theta[1]*age_norm[i] + theta[2]*sofa_norm[i]
            g_exp = theta[3] + theta[4]*age_norm[i] + theta[5]*sofa_norm[i]

            params_i = optimizeIndividualParams_regularized(
                L_obs[i, :], A_obs[i, :], L0_est[i, :], theta[6], dt,
                expected_params=np.array([C_exp, g_exp], float),
                reg_weight=reg_weights[i],
            )
            C_indiv[i], g_indiv[i] = params_i[0], params_i[1]

        # 2) Ridge updates for population coefs
        valid_C = np.isfinite(C_indiv) & (C_indiv > 0)
        if np.sum(valid_C) > 10:
            X = np.column_stack([np.ones(np.sum(valid_C)), age_norm[valid_C], sofa_norm[valid_C]])
            lamC = RegularizationStrength * 10.0
            beta_C = _ridge_closed_form(X, C_indiv[valid_C], lamC, prior_means[:3])
            theta[0:3] = momentum * theta[0:3] + (1 - momentum) * beta_C
            C_pred = X @ theta[0:3]
            theta[7] = float(np.sqrt(np.mean((C_indiv[valid_C] - C_pred) ** 2)))

        valid_g = np.isfinite(g_indiv) & (g_indiv > 0)
        if np.sum(valid_g) > 10:
            Xg = np.column_stack([np.ones(np.sum(valid_g)), age_norm[valid_g], sofa_norm[valid_g]])
            lamG = RegularizationStrength * 10.0
            beta_g = _ridge_closed_form(Xg, g_indiv[valid_g], lamG, prior_means[3:6])
            theta[3:6] = momentum * theta[3:6] + (1 - momentum) * beta_g
            g_pred = Xg @ theta[3:6]
            theta[8] = float(np.sqrt(np.mean((g_indiv[valid_g] - g_pred) ** 2)))

        # 3) Update ke with regularization toward prior
        ke_new = optimizeKe_regularized(
            L_obs, A_obs, L0_est, C_indiv, g_indiv, dt,
            prior_ke=prior_means[6], reg_strength=RegularizationStrength
        )
        theta[6] = momentum * theta[6] + (1 - momentum) * ke_new

        # Shrink individuals toward updated population
        C_pop = theta[0] + theta[1]*age_norm + theta[2]*sofa_norm
        g_pop = theta[3] + theta[4]*age_norm + theta[5]*sofa_norm
        shrink = 0.3
        C_indiv = (1 - shrink) * C_indiv + shrink * C_pop
        g_indiv = (1 - shrink) * g_indiv + shrink * g_pop
        C_indiv = _clip(C_indiv, 0.5, 10.0)
        g_indiv = _clip(g_indiv, 0.5, 10.0)

        theta_history.append(theta.copy())

        # Convergence checks
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

    # Outputs
    theta_est = theta.copy()

    patient_params = {
        "C_indiv": C_indiv,
        "g_indiv": g_indiv,
        "C_pred": theta[0] + theta[1]*age_norm + theta[2]*sofa_norm,
        "g_pred": theta[3] + theta[4]*age_norm + theta[5]*sofa_norm,
    }

    results: Dict[str, object] = {}
    results["iterations"] = iter_
    results["log_likelihood_history"] = np.asarray(log_likelihood_history, float)
    results["theta_history"] = np.asarray(theta_history, float).T if theta_history else np.empty((9, 0))
    results["converged"] = (iter_ < MaxIterEM)
    results["data_quality"] = data_quality

    # R² diagnostics
    def _r2(y, yhat):
        if np.sum(np.isfinite(y)) <= 10:
            return np.nan
        yv = y[np.isfinite(y)]
        yhv = yhat[np.isfinite(y)]
        num = np.sum((yv - yhv) ** 2)
        den = np.sum((yv - np.mean(yv)) ** 2) + 1e-12
        return 1.0 - (num / den)

    results["R2_C"] = _r2(C_indiv, patient_params["C_pred"])
    results["R2_g"] = _r2(g_indiv, patient_params["g_pred"])

    if Verbose:
        print("\n========================================")
        print("FINAL ESTIMATES:")
        print(f"C = {theta[0]:.3f} + {theta[1]:.3f}*age + {theta[2]:.3f}*sofa (σ={theta[7]:.3f})")
        print(f"g = {theta[3]:.3f} + {theta[4]:.3f}*age + {theta[5]:.3f}*sofa (σ={theta[8]:.3f})")
        print(f"ke = {theta[6]:.3f}")
        print(f"R² for C: {results['R2_C']:.3f}, R² for g: {results['R2_g']:.3f}")
        print("========================================")

    return theta_est, patient_params, L0_est, results


# ---------------------------------------------------------------------
# Helpers (translated MATLAB subfunctions)
# ---------------------------------------------------------------------

def assessDataQuality(L_obs: np.ndarray, A_obs: np.ndarray) -> np.ndarray:
    """Quality per patient in [0.1, 1], higher = better."""
    N, T = L_obs.shape
    q = np.zeros(N, float)
    for i in range(N):
        completeness = np.mean(np.isfinite(L_obs[i, :]))
        A_nonzero = A_obs[i, A_obs[i, :] > 0]
        if A_nonzero.size > 0 and np.mean(A_nonzero) > 0:
            dose_cv = float(np.std(A_nonzero, ddof=1) / np.mean(A_nonzero))
            dose_term = min(dose_cv, 1.0)
        else:
            dose_term = 0.0
        valid_L = L_obs[i, np.isfinite(L_obs[i, :])]
        if valid_L.size > 10:
            ac = np.corrcoef(valid_L[:-1], valid_L[1:])[0, 1]
            autocorr_val = float(abs(ac)) if np.isfinite(ac) else 0.0
        else:
            autocorr_val = 0.0
        q[i] = completeness * (0.5 + 0.5 * dose_term) * autocorr_val
    # Normalize to [0.1, 1]
    q_min, q_max = float(np.min(q)), float(np.max(q))
    if q_max - q_min < 1e-12:
        return 0.1 + 0.9 * np.ones_like(q)
    return 0.1 + 0.9 * (q - q_min) / (q_max - q_min)


def computeRegularizationWeights(data_quality: np.ndarray, base_strength: float) -> np.ndarray:
    """Higher regularization for lower quality data; capped at 10."""
    w = base_strength / (data_quality + 0.1)
    return np.minimum(10.0, w)


def computePriorLikelihood(theta: np.ndarray, prior_means: np.ndarray, prior_stds: np.ndarray) -> float:
    z = (np.asarray(theta) - np.asarray(prior_means)) / np.asarray(prior_stds)
    return float(-0.5 * np.sum(z ** 2))


def initializeL0_improved(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    C: np.ndarray,
    g: np.ndarray,
    ke: float,
    parmsL: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Inverse treatment, interpolate NaNs, robust-smooth, and bound by 2*peak_height."""
    N, T = L_obs.shape
    L0 = L_obs.copy()
    peak_height = float(parmsL[1]) if parmsL.size > 1 else 1.0

    X = np.zeros(T, float)
    for i in range(N):
        X[:] = 0.0
        for j in range(1, T):
            X[j] = ke * X[j-1] + A_obs[i, j]
            if (X[j-1] > 0) and np.isfinite(L_obs[i, j]):
                sX = 1.0 - 1.0 / ((C[i] / X[j-1]) ** g[i] + 1.0)
                if sX > 0.1:
                    L0[i, j] = L_obs[i, j] / sX

        # Fill missing by linear interp over indices
        valid = np.isfinite(L0[i, :])
        if np.sum(valid) > 3:
            idx = np.arange(T)
            L0[i, ~valid] = np.interp(idx[~valid], idx[valid], L0[i, valid])

        # Robust-ish smoothing (moving average as a stand-in for rloess)
        L0[i, :] = _smooth_1d(L0[i, :], window=7)
        L0[i, :] = _clip(L0[i, :], 0.0, 2.0 * peak_height)

    return L0


def runEKF_improved(
    L_i: np.ndarray,
    A_i: np.ndarray,
    C: float,
    g: float,
    ke: float,
    parmsL: np.ndarray,
    dt: float,
    Q: float,
    R: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Improved EKF with adaptive measurement noise and slow-evolving L0 dynamics."""
    T = L_i.size
    L0_filt = np.zeros(T, float)
    P_filt = np.zeros(T, float)

    # Initial state
    if A_i[0] == 0 or not np.isfinite(A_i[0]):
        L0_filt[0] = float(L_i[0]) if np.isfinite(L_i[0]) else 0.5 * float(parmsL[1])
    else:
        L0_filt[0] = 0.5 * float(parmsL[1])
    P_filt[0] = 0.1

    X = np.zeros(T, float)
    log_lik = 0.0

    for t_ in range(1, T):
        X[t_] = ke * X[t_-1] + (A_i[t_] if np.isfinite(A_i[t_]) else 0.0)

        # Predict: L0 tends mildly toward peak height
        L0_pred = L0_filt[t_-1] * (1.0 - 0.01*dt) + 0.01 * float(parmsL[1]) * dt
        P_pred = P_filt[t_-1] + Q

        if np.isfinite(L_i[t_]):
            if X[t_-1] > 0:
                sX = 1.0 - 1.0 / ((C / X[t_-1]) ** g + 1.0)
                H = sX
            else:
                sX = 1.0
                H = 1.0

            y_pred = L0_pred * sX
            innov = L_i[t_] - y_pred

            R_adapt = R * (1.0 + abs(innov) / (abs(y_pred) + 0.1))
            S = H * P_pred * H + R_adapt
            K = (P_pred * H) / S

            L0_filt[t_] = L0_pred + K * innov
            P_filt[t_] = (1.0 - K * H) * P_pred

            if S > 0:
                log_lik -= 0.5 * (np.log(2.0 * np.pi * S) + (innov ** 2) / S)
        else:
            L0_filt[t_] = L0_pred
            P_filt[t_] = P_pred

    # No separate backward RTS step; light smoothing
    L0_smooth = _smooth_1d(L0_filt, window=5)
    L0_smooth = _clip(L0_smooth, 0.0, np.inf)
    return L0_smooth, P_filt, float(log_lik)


def optimizeIndividualParams_regularized(
    L_i: np.ndarray,
    A_i: np.ndarray,
    L0_i: np.ndarray,
    ke: float,
    dt: float,
    expected_params: np.ndarray,
    reg_weight: float
) -> np.ndarray:
    """Optimize C,g with L2 regularization (toward expected_params)."""
    def obj(p):
        return individualNLL_regularized(p, L_i, A_i, L0_i, ke, expected_params, reg_weight)

    bounds = [(0.5, 10.0), (0.5, 10.0)]
    res = minimize(obj, x0=np.asarray(expected_params, float), method="SLSQP",
                   bounds=bounds, options={"maxiter": 150, "ftol": 1e-9, "disp": False})
    return res.x.astype(float)


def individualNLL_regularized(
    params: np.ndarray,
    L_i: np.ndarray,
    A_i: np.ndarray,
    L0_i: np.ndarray,
    ke: float,
    expected_params: np.ndarray,
    reg_weight: float
) -> float:
    """NLL for one patient with L2 regularization; variance increases with level."""
    C, g = float(params[0]), float(params[1])
    T = L_i.size

    X = np.zeros(T, float)
    nll = 0.0
    for t_ in range(1, T):
        X[t_] = ke * X[t_ - 1] + (A_i[t_] if np.isfinite(A_i[t_]) else 0.0)
        if np.isfinite(L_i[t_]) and (L0_i[t_] > 0):
            if X[t_-1] > 0:
                sX = 1.0 - 1.0 / ((C / X[t_-1]) ** g + 1.0)
                L_pred = L0_i[t_] * sX
            else:
                L_pred = L0_i[t_]
            var_L = 0.01 + 0.05 * float(L_pred)
            if var_L > 0 and L_pred >= 0:
                nll += 0.5 * np.log(2.0*np.pi*var_L) + 0.5 * ((L_i[t_] - L_pred) ** 2) / var_L

    # L2 regularization
    nll += reg_weight * float(np.sum((np.asarray(params) - np.asarray(expected_params)) ** 2))
    return float(nll)


def optimizeKe_regularized(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    L0_est: np.ndarray,
    C_indiv: np.ndarray,
    g_indiv: np.ndarray,
    dt: float,
    prior_ke: float,
    reg_strength: float
) -> float:
    """Optimize ke in [0.1, 1.0] with penalty toward prior_ke."""
    def obj(ke_scalar):
        ke_val = float(ke_scalar[0]) if np.ndim(ke_scalar) else float(ke_scalar)
        return totalNLL_ke_regularized(
            ke_val, L_obs, A_obs, L0_est, C_indiv, g_indiv, prior_ke, reg_strength
        )

    bounds = [(0.1, 1.0)]
    res = minimize(obj, x0=np.array([prior_ke], float), method="SLSQP",
                   bounds=bounds, options={"maxiter": 100, "ftol": 1e-8, "disp": False})
    return float(res.x[0])


def totalNLL_ke_regularized(
    ke: float,
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    L0_est: np.ndarray,
    C_indiv: np.ndarray,
    g_indiv: np.ndarray,
    prior_ke: float,
    reg_strength: float
) -> float:
    """Total NLL across treated patients + quadratic penalty to prior_ke."""
    N, T = L_obs.shape
    nll = 0.0
    for i in range(N):
        if np.nanmax(A_obs[i, :]) > 0:
            X = np.zeros(T, float)
            for t_ in range(1, T):
                X[t_] = ke * X[t_-1] + (A_obs[i, t_] if np.isfinite(A_obs[i, t_]) else 0.0)
                if np.isfinite(L_obs[i, t_]) and (L0_est[i, t_] > 0):
                    if X[t_-1] > 0:
                        sX = 1.0 - 1.0 / ((C_indiv[i] / X[t_-1]) ** g_indiv[i] + 1.0)
                        L_pred = L0_est[i, t_] * sX
                    else:
                        L_pred = L0_est[i, t_]
                    var_L = 0.01 + 0.05 * float(L_pred)
                    if var_L > 0 and L_pred >= 0:
                        nll += 0.5 * np.log(2.0*np.pi*var_L) + 0.5 * ((L_obs[i, t_] - L_pred) ** 2) / var_L
    # Regularization toward prior ke
    nll += reg_strength * 100.0 * (ke - prior_ke) ** 2
    return float(nll)


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float, prior_means_3: np.ndarray) -> np.ndarray:
    XT = X.T
    A = XT @ X + lam * np.eye(3)
    b = XT @ y + lam * np.asarray(prior_means_3, float)
    return np.linalg.solve(A, b)

def _smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, float)
    w = int(max(1, window))
    if x.size < w or w == 1:
        return x.copy()
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, float) / w
    return np.convolve(xpad, kernel, mode="valid")

def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(arr, float), lo, hi)

def _rel_change(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a.ravel())
    denom = na + 1e-12
    return float(np.linalg.norm((a - b).ravel()) / denom)
