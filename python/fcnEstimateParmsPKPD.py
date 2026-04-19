# fcnEstimateParmsPKPD.py
# Estimate PD parameters given fixed L parameters

import numpy as np
from typing import Tuple
from scipy.optimize import minimize

def fcnEstimateParmsPKPD(
    parmsL_est: np.ndarray,
    LL: np.ndarray,
    AA: np.ndarray,
    patient_age: np.ndarray,
    patient_sofa: np.ndarray,
    t: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python translation of:
      [ke_est, C_est_all, g_est_all, theta_est] = fcnEstimateParmsPKPD(parmsL_est, LL, AA, patient_age, patient_sofa, t)

    Inputs
    ------
    parmsL_est   : (7,) array [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
    LL, AA       : (N_patients, Nt) arrays (NaN-padded trajectories)
    patient_age  : (N_patients,)
    patient_sofa : (N_patients,)
    t            : (Nt,) time grid (hours)

    Returns
    -------
    ke_est       : float
    C_est_all    : (N_patients,) array
    g_est_all    : (N_patients,) array
    theta_est    : (7 + 6 + 1,) array = [parmsL_est(:); b0_C; b1_C; b2_C; b0_g; b1_g; b2_g; ke]
    """
    parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)
    LL = np.asarray(LL, dtype=float)
    AA = np.asarray(AA, dtype=float)
    patient_age = np.asarray(patient_age, dtype=float).reshape(-1)
    patient_sofa = np.asarray(patient_sofa, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    dt = float(t[1] - t[0])
    n_patients, Nt = LL.shape

    # Identify treated patients (any A > 0 across time)
    is_treated = np.nanmax(AA, axis=1) > 0
    n_treated = int(np.sum(is_treated))
    if n_treated == 0:
        raise ValueError("No treated patients - cannot estimate PD parameters")

    # Normalization (match MATLAB's mean/std; ddof=1)
    age_mean = np.mean(patient_age); age_std = np.std(patient_age, ddof=1) or 1.0
    sofa_mean = np.mean(patient_sofa); sofa_std = np.std(patient_sofa, ddof=1) or 1.0

    age_norm_all = (patient_age - age_mean) / age_std
    sofa_norm_all = (patient_sofa - sofa_mean) / sofa_std

    age_treated = patient_age[is_treated]
    sofa_treated = patient_sofa[is_treated]
    # Use SAME population mean/std as MATLAB code does
    age_norm_treated = (age_treated - age_mean) / age_std
    sofa_norm_treated = (sofa_treated - sofa_mean) / sofa_std

    L_treated = LL[is_treated, :]
    A_treated = AA[is_treated, :]

    # Initial guess: [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke]
    theta_PD_init = np.array([3.0, 0.1, 0.15, 4.0, 0.08, 0.12, 0.75], dtype=float)

    # Bounds (lb, ub) as in MATLAB
    lb_PD = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1], dtype=float)
    ub_PD = np.array([10.0, 0.5, 0.5, 8.0, 0.5, 0.5, 0.9], dtype=float)
    bounds = list(zip(lb_PD, ub_PD))

    def nll_PD(theta_PD: np.ndarray) -> float:
        return _compute_PD_regression_nll(
            theta_PD, parmsL_est, L_treated, A_treated,
            age_norm_treated, sofa_norm_treated, t, dt
        )

    # Optimization (box constraints)
    res = minimize(
        nll_PD, theta_PD_init, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-9}
    )
    theta_PD_est = res.x.astype(float)

    # Extract estimated parameters
    b0_C_est, b1_C_est, b2_C_est, b0_g_est, b1_g_est, b2_g_est, ke_est = theta_PD_est.tolist()

    # Individual C and g values for ALL patients using population model
    C_est_all = b0_C_est + b1_C_est * age_norm_all + b2_C_est * sofa_norm_all
    g_est_all = b0_g_est + b1_g_est * age_norm_all + b2_g_est * sofa_norm_all

    # Combine all parameters
    theta_est = np.concatenate([
        parmsL_est.reshape(-1),
        np.array([b0_C_est, b1_C_est, b2_C_est, b0_g_est, b1_g_est, b2_g_est, ke_est], dtype=float)
    ])

    return ke_est, C_est_all, g_est_all, theta_est


# =====================================================================
# Helper(s) — mirrors MATLAB compute_PD_regression_nll
# =====================================================================
def _compute_PD_regression_nll(
    theta_PD: np.ndarray,
    L_params: np.ndarray,
    L_treated: np.ndarray,
    A_treated: np.ndarray,
    age_norm: np.ndarray,
    sofa_norm: np.ndarray,
    t: np.ndarray,
    dt: float,
) -> float:
    """
    Negative log-likelihood for PD regression parameters with fixed L parameters.
    theta_PD = [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke]
    """
    b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke = [float(v) for v in theta_PD]
    N_treated, Nt = L_treated.shape

    nll = 0.0

    # Generate deterministic natural history L0 based on L_params (single trajectory),
    # matching MATLAB's fcn_generateTrajectory intent.
    L0 = _generate_L0_deterministic(L_params, t)  # shape (Nt,)

    for i in range(N_treated):
        L = L_treated[i, :]
        A = A_treated[i, :]

        # Patient-specific C and g
        C_i = b0_C + b1_C * age_norm[i] + b2_C * sofa_norm[i]
        g_i = b0_g + b1_g * age_norm[i] + b2_g * sofa_norm[i]
        C_i = max(0.1, C_i)
        g_i = max(0.1, g_i)

        # Drug concentration state for this patient
        x = np.zeros(Nt, dtype=float)

        for j in range(1, Nt):
            Lj = L[j]
            Ajm1 = A[j - 1]
            if not np.isfinite(Lj):
                continue
            if Lj < 0:
                continue
            # Expected L given previous concentration x[j-1]
            if x[j - 1] > 0:
                sX = 1.0 - 1.0 / ((C_i / x[j - 1]) ** g_i + 1.0)
                L_expected = L0[j] * sX
            else:
                L_expected = L0[j]

            # Concentration dynamics (discrete, per MATLAB)
            x[j] = ke * x[j - 1] + A[j]

            # Heteroskedastic variance (treated has larger scale)
            if Ajm1 > 0:
                var_L = 0.01 + 0.1 * L_expected + 0.05 * (L_expected ** 2)
            else:
                var_L = 0.001 + 0.05 * L_expected + 0.01 * (L_expected ** 2)

            if var_L > 0 and L_expected >= 0:
                # Gaussian log-likelihood contribution (include normalizing constant)
                ll = _log_normpdf(Lj, L_expected, np.sqrt(var_L))
                if np.isfinite(ll):
                    nll -= ll

    # Light L2 regularization (as in MATLAB)
    nll += 0.01 * float(np.sum(np.square(theta_PD)))

    if not np.isfinite(nll):
        nll = 1e10
    return nll


def _generate_L0_deterministic(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Deterministic natural-history trajectory using the SAME drift as your
    stochastic generator, but without diffusion (noise). This mirrors the
    intent of MATLAB's fcn_generateTrajectory called in the PD objective.
    params = [gamma, H, alpha, delta, sigma_early, sigma_late, tau]
    """
    gamma, H, alpha, delta = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    # sigmas/tau are not used (no diffusion)
    Nt = t.size
    X = np.zeros(Nt, dtype=float)
    # X[0] = 0 by MATLAB convention
    for j in range(1, Nt):
        current_time = t[j]
        x_prev = X[j - 1]
        if current_time < 30.0:
            growth_term = gamma * x_prev * (1.0 - x_prev / H)
        else:
            time_since_peak = current_time - 30.0
            decay_factor = np.exp(-delta * time_since_peak)
            growth_term = -alpha * (x_prev - H * decay_factor * 0.2)
        mean_reversion = -alpha * max(0.0, x_prev - H)
        drift = growth_term + mean_reversion
        # Euler step (dt is uniform based on t vector)
        dt = t[j] - t[j - 1]
        X[j] = max(0.0, x_prev + drift * dt)
    return X  # shape (Nt,)


def _log_normpdf(x: float, mu: float, sigma: float) -> float:
    """Log of univariate normal pdf with mean mu and std sigma."""
    if sigma <= 0 or not np.isfinite(x) or not np.isfinite(mu) or not np.isfinite(sigma):
        return -np.inf
    z = (x - mu) / sigma
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)
