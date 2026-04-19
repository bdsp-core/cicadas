# fcnEstimateParmsL.py

import numpy as np
import pandas as pd
from typing import Tuple

# This helper must exist and return (parmsL_est, diagnostics_L),
# where parmsL_est is length-7 (gamma, H, alpha, delta, sigma_early, sigma_late, tau).
from fcn_estimate_parmsL import fcn_estimate_parmsL  # noqa: F401

def fcnEstimateParmsL(T0: pd.DataFrame):
    """
    Estimate disease-model L parameters from UNTREATED patients.

    Mirrors MATLAB:
      [parmsL_est, LL, AA, patient_age, patient_sofa, t] = fcnEstimateParmsL(T0)

    Parameters
    ----------
    T0 : DataFrame
        Must contain columns: ['sid','t','age','sofa','L','A','Rx'].

    Returns
    -------
    parmsL_est : (7,) np.ndarray
    LL         : (N_patients, Nt) np.ndarray (NaN-padded)
    AA         : (N_patients, Nt) np.ndarray (NaN-padded)
    patient_age : (N_patients,) np.ndarray
    patient_sofa: (N_patients,) np.ndarray
    t           : (Nt,) np.ndarray  time grid (0:2:168)
    """
    required = {"sid", "t", "age", "sofa", "L", "A", "Rx"}
    missing = required - set(T0.columns)
    if missing:
        raise ValueError(f"T0 is missing required columns: {sorted(missing)}")

    # Regular time grid (MATLAB: dt=2; t=0:dt:168;)
    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)
    Nt = t.size  # 85

    # One row per unique patient (first occurrence like MATLAB find(...,1,'first'))
    unique_patients = np.unique(T0["sid"].values)
    n_patients = unique_patients.size

    patient_age = np.zeros(n_patients, dtype=float)
    patient_sofa = np.zeros(n_patients, dtype=float)
    patient_treated = np.zeros(n_patients, dtype=float)  # parity with MATLAB (not used)

    for i, pid in enumerate(unique_patients):
        # First index where sid == pid, preserving file order
        first_idx = T0.index[T0["sid"].values == pid][0]
        patient_age[i] = float(T0.at[first_idx, "age"])
        patient_sofa[i] = float(T0.at[first_idx, "sofa"])
        patient_treated[i] = float(T0.at[first_idx, "Rx"])

    # Trajectory matrices (NaN-padded to the common grid)
    LL = np.full((n_patients, Nt), np.nan, dtype=float)
    AA = np.full((n_patients, Nt), np.nan, dtype=float)

    for i, pid in enumerate(unique_patients):
        pdata = T0[T0["sid"] == pid]  # keep input order, like MATLAB
        n_obs = len(pdata)
        L_vals = pdata["L"].to_numpy(dtype=float)
        A_vals = pdata["A"].to_numpy(dtype=float)
        m = min(n_obs, Nt)
        LL[i, :m] = L_vals[:m]
        AA[i, :m] = A_vals[:m]

    # Identify untreated patients (any A>0 across time => treated)
    is_treated = np.nanmax(AA, axis=1) > 0
    n_untreated = int(np.sum(~is_treated))
    if n_untreated < 10:
        raise ValueError("Need at least 10 untreated patients for L parameter estimation")

    # Estimate L parameters using ONLY untreated patients
    L_untreated = LL[~is_treated, :]
    parmsL_est, diagnostics_L = fcn_estimate_parmsL(L_untreated, t)

    # MATLAB ends with parmsL_est = parmsL_est'; here we return 1D
    parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)

    return parmsL_est, LL, AA, patient_age, patient_sofa, t
