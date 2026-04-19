# fcnGetPKPD_parms_est.py

import numpy as np
from scipy.io import loadmat

def fcnGetPKPD_parms_est(patient_age: np.ndarray, patient_sofa: np.ndarray):
    """
    Python translation of fcnGetPKPD_parms_est.m

    Inputs
    ------
    patient_age : (N,) array
    patient_sofa: (N,) array

    Returns
    -------
    C_est      : (N,) array
    g_est      : (N,) array
    ke_est     : float scalar (two-stage corrected)
    parmsPD_est: (6,) array [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g]
    """
    # Load the results struct saved by your PKPD estimation pipeline
    # NOTE: This expects a v7 .mat (the default saved by scipy.io.savemat or MATLAB without -v7.3).
    # If your file was saved in MATLAB with '-v7.3', loadmat won't work—resave without -v7.3
    # or implement an h5py loader.
    m = loadmat("PKPD_estimation_results.mat", squeeze_me=True, struct_as_record=False)
    results = m.get("results", None)
    if results is None:
        raise FileNotFoundError("Could not find 'results' in PKPD_estimation_results.mat")

    # Access nested fields: results.twostage_corr
    # With struct_as_record=False and squeeze_me=True, MATLAB structs come in as 'mat_struct' objects.
    twostage_corr = getattr(results, "twostage_corr", None)
    if twostage_corr is None:
        raise KeyError("Missing 'twostage_corr' in results")

    # ke (bias-corrected)
    if hasattr(twostage_corr, "ke_corrected"):
        ke_est = float(np.asarray(twostage_corr.ke_corrected).squeeze())
    else:
        # Fallback if only raw ke present
        ke_est = float(np.asarray(getattr(twostage_corr, "ke_raw")).squeeze())

    # Mixed-effects coefficients (theta): [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, (ke?)]
    theta = np.asarray(getattr(twostage_corr, "theta")).astype(float).squeeze()
    parmsPD_est = theta[:6].copy()  # ensure exactly 6 (no ke)

    # z-score normalization matching MATLAB: std with N-1 (ddof=1)
    patient_age = np.asarray(patient_age, dtype=float).reshape(-1)
    patient_sofa = np.asarray(patient_sofa, dtype=float).reshape(-1)
    if patient_age.size != patient_sofa.size:
        raise ValueError("patient_age and patient_sofa must have the same length")

    age_mean = np.mean(patient_age)
    sofa_mean = np.mean(patient_sofa)
    # ddof=1 to match MATLAB std()
    age_std = np.std(patient_age, ddof=1)
    sofa_std = np.std(patient_sofa, ddof=1)
    # guard against zero variance
    if age_std == 0:
        age_std = 1.0
    if sofa_std == 0:
        sofa_std = 1.0

    age_norm = (patient_age - age_mean) / age_std
    sofa_norm = (patient_sofa - sofa_mean) / sofa_std

    # Individual C and g from population model
    b0_C, b1_C, b2_C, b0_g, b1_g, b2_g = parmsPD_est.tolist()
    C_est = b0_C + b1_C * age_norm + b2_C * sofa_norm
    g_est = b0_g + b1_g * age_norm + b2_g * sofa_norm

    return C_est, g_est, ke_est, parmsPD_est
