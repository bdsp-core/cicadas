# %% a5_EstimatePKPD.py
# Complete PKPD Parameter Estimation Pipeline
# This script runs all estimation methods and saves results for evaluation
#
# Outputs saved to: PKPD_estimation_results.mat

import time
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

# Optional: mirror MATLAB's rng(0) for reproducibility of any randomness inside estimators
np.random.seed(0)

print("==========================================================")
print("a5_EstimatePKPD: COMPLETE PARAMETER ESTIMATION PIPELINE")
print("==========================================================")
print("This will run all methods and save results for plotting\n")

# ------------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ------------------------------------------------------------
print("[1/6] Loading and preparing data...")
tic = time.perf_counter()

# Load data
T = pd.read_csv("trialDataDoseChanging.csv")
m = loadmat("parmsTrue_DoseChanging.mat", squeeze_me=True, struct_as_record=False)
# Expect: 'parmsPD', 'C', 'g', 'ke', 'age', 'sofa', 'parmsL'
parmsPD = np.asarray(m["parmsPD"]).astype(float).reshape(-1)
C = np.asarray(m["C"]).astype(float).reshape(-1)
g = np.asarray(m["g"]).astype(float).reshape(-1)
ke = float(np.asarray(m["ke"]).reshape(()))
age = np.asarray(m["age"]).astype(float).reshape(-1)
sofa = np.asarray(m["sofa"]).astype(float).reshape(-1)
parmsL = np.asarray(m["parmsL"]).astype(float).reshape(-1)

L0mat = loadmat("L0data.mat", squeeze_me=True, struct_as_record=False)
# key may be 'L0' depending on how it was saved
L0_true = np.asarray(L0mat["L0"]).astype(float)

# Prepare data matrices
unique_patients = np.unique(T["sid"].values)
N = unique_patients.size
dt = 2
t = np.arange(0, 168 + dt, dt)  # 0:dt:168 inclusive
T_len = t.size

L_obs = np.full((N, T_len), np.nan)
A_obs = np.zeros((N, T_len), dtype=float)
patient_age = np.zeros((N, 1), dtype=float)
patient_sofa = np.zeros((N, 1), dtype=float)
C_true = np.zeros((N, 1), dtype=float)
g_true = np.zeros((N, 1), dtype=float)

# Map each patient's rows into row i at the proper time indices
sid_to_row = {sid: i for i, sid in enumerate(unique_patients)}
for sid in unique_patients:
    i = sid_to_row[sid]
    pdata = T[T["sid"] == sid].sort_values("t")
    # MATLAB: time_idx = patient_data.t / dt + 1; -> 1-based
    time_idx0 = (pdata["t"].values / dt).astype(int)  # 0-based indices
    L_obs[i, time_idx0] = pdata["L"].values
    A_obs[i, time_idx0] = pdata["A"].values
    patient_age[i, 0] = float(pdata["age"].iloc[0])
    patient_sofa[i, 0] = float(pdata["sofa"].iloc[0])
    C_true[i, 0] = float(C[i])
    g_true[i, 0] = float(g[i])

# Store true parameters
true_params = np.concatenate([parmsPD.reshape(-1), np.array([ke], dtype=float)])
param_names = ["b0_C", "b1_C", "b2_C", "b0_g", "b1_g", "b2_g", "ke"]

time_prep = time.perf_counter() - tic
print(f"  Data loaded: {N} patients, {T_len} time points each")
print(f"  True ke = {ke:.3f}")
print(f"  Time: {time_prep:.1f} seconds\n")

# ------------------------------------------------------------
# External estimator functions (translate & provide these)
# ------------------------------------------------------------
# Joint estimation (unfixed ke)
from fcnEstimatePKPD_StateSpaceMixedEffects_v2 import fcnEstimatePKPD_StateSpaceMixedEffects_v2
# Fixed-ke optimized estimator
from fcnEstimatePKPD_FixedKe_Optimized import fcnEstimatePKPD_FixedKe_Optimized
# Standalone ke estimation (raw)
from fcnEstimateKe_Standalone import fcnEstimateKe_Standalone
# ke estimation with bias correction
from fcn_EstimateKe_WithBiasCorrection import fcnEstimateKe_WithBiasCorrection

# ------------------------------------------------------------
# Helper: computeL0Metrics (translation of nested MATLAB function)
# ------------------------------------------------------------
def computeL0Metrics(L0_est: np.ndarray, L0_true_: np.ndarray):
    """
    Returns: mean_corr, mean_rmse, mean_mae
    Computed across patients where there are >10 valid points.
    """
    Np = L0_est.shape[0]
    corr_vals = np.zeros(Np)
    rmse_vals = np.zeros(Np)
    mae_vals = np.zeros(Np)

    for i in range(Np):
        valid = np.isfinite(L0_est[i, :]) & np.isfinite(L0_true_[i, :])
        if np.sum(valid) > 10:
            x = L0_est[i, valid]
            y = L0_true_[i, valid]
            # correlation
            if np.std(x) > 0 and np.std(y) > 0:
                corr_vals[i] = np.corrcoef(x, y)[0, 1]
            else:
                corr_vals[i] = 0.0
            # RMSE / MAE
            diff = x - y
            rmse_vals[i] = np.sqrt(np.mean(diff ** 2))
            mae_vals[i] = np.mean(np.abs(diff))
        else:
            corr_vals[i] = 0.0
            rmse_vals[i] = np.nan
            mae_vals[i] = np.nan

    valid_idx = corr_vals > 0
    mean_corr = float(np.nanmean(corr_vals[valid_idx])) if np.any(valid_idx) else 0.0
    mean_rmse = float(np.nanmean(rmse_vals[valid_idx])) if np.any(valid_idx) else np.nan
    mean_mae = float(np.nanmean(mae_vals[valid_idx])) if np.any(valid_idx) else np.nan
    return mean_corr, mean_rmse, mean_mae

# ------------------------------------------------------------
# 2. METHOD 1: JOINT ESTIMATION (Unfixed ke)
# ------------------------------------------------------------
print("[2/6] Running Joint Estimation (all parameters)...")
tic = time.perf_counter()

(theta_joint,
 patient_params_joint,
 L0_joint,
 results_joint) = fcnEstimatePKPD_StateSpaceMixedEffects_v2(
    L_obs, A_obs, patient_age, patient_sofa, t, parmsL,
    RegularizationStrength=5.0,
    UsePriors=True,
    PriorMeans=true_params,
    PriorStds=np.array([0.5, 0.05, 0.05, 0.5, 0.05, 0.05, 0.1]),
    MaxIterEM=30,
    Verbose=False
)

time_joint = time.perf_counter() - tic
errors_joint = np.abs(theta_joint[:7] - true_params) / np.abs(true_params) * 100.0
mape_joint = float(np.mean(errors_joint))
corr_joint, rmse_joint, mae_joint = computeL0Metrics(L0_joint, L0_true)

print(f"  MAPE: {mape_joint:.1f}%, L0 corr: {corr_joint:.3f}")
print(f"  Time: {time_joint:.1f} seconds\n")

# ------------------------------------------------------------
# 3. METHOD 2: ORACLE (Fixed ke with true value)
# ------------------------------------------------------------
print("[3/6] Running Oracle Estimation (fixed true ke)...")
tic = time.perf_counter()

(theta_oracle,
 patient_params_oracle,
 L0_oracle,
 results_oracle) = fcnEstimatePKPD_FixedKe_Optimized(
    L_obs, A_obs, patient_age, patient_sofa, t, parmsL, ke,
    RegularizationStrength=5.0,
    UsePriors=True,
    PriorMeans=true_params[:6],
    PriorStds=np.array([0.5, 0.05, 0.05, 0.5, 0.05, 0.05]),
    MaxIterEM=30,
    Verbose=False
)

time_oracle = time.perf_counter() - tic
theta_oracle_full = np.concatenate([theta_oracle[:6], np.array([ke])])
errors_oracle = np.abs(theta_oracle_full - true_params) / np.abs(true_params) * 100.0
mape_oracle = float(np.mean(errors_oracle))
corr_oracle, rmse_oracle, mae_oracle = computeL0Metrics(L0_oracle, L0_true)

print(f"  MAPE: {mape_oracle:.1f}%, L0 corr: {corr_oracle:.3f}")
print(f"  Time: {time_oracle:.1f} seconds\n")

# ------------------------------------------------------------
# 4. METHOD 3: TWO-STAGE WITHOUT BIAS CORRECTION
# ------------------------------------------------------------
print("[4/6] Running Two-Stage Estimation (no correction)...")
tic = time.perf_counter()

# Stage 1: Estimate ke (raw)
ke_raw, results_ke_raw = fcnEstimateKe_Standalone(
    L_obs, A_obs,
    AssumeC=float(np.mean(C)),
    AssumeG=float(np.mean(g)),
    KeRange=np.array([0.3, 0.9]),
    Verbose=False
)

print(f"  Stage 1 - Raw ke: {ke_raw:.3f} (error: {abs(ke_raw - ke)/ke*100:.1f}%)")

# Stage 2: Fix ke and estimate others
(theta_twostage_raw,
 patient_params_raw,
 L0_twostage_raw,
 results_twostage_raw) = fcnEstimatePKPD_FixedKe_Optimized(
    L_obs, A_obs, patient_age, patient_sofa, t, parmsL, ke_raw,
    MaxIterEM=20,
    Verbose=False
)

time_twostage_raw = time.perf_counter() - tic
theta_twostage_raw_full = np.concatenate([theta_twostage_raw[:6], np.array([ke_raw])])
errors_twostage_raw = np.abs(theta_twostage_raw_full - true_params) / np.abs(true_params) * 100.0
mape_twostage_raw = float(np.mean(errors_twostage_raw))
corr_twostage_raw, rmse_twostage_raw, mae_twostage_raw = computeL0Metrics(L0_twostage_raw, L0_true)

print(f"  Overall MAPE: {mape_twostage_raw:.1f}%, L0 corr: {corr_twostage_raw:.3f}")
print(f"  Time: {time_twostage_raw:.1f} seconds\n")

# ------------------------------------------------------------
# 5. METHOD 4: TWO-STAGE WITH BIAS CORRECTION
# ------------------------------------------------------------
print("[5/6] Running Two-Stage Estimation (with correction)...")
tic = time.perf_counter()

# Stage 1: Estimate ke with bias correction
(ke_corrected,
 ke_raw2,
 results_ke_corrected) = fcnEstimateKe_WithBiasCorrection(
    L_obs, A_obs,
    CorrectionFactor=1.41,
    UsePrior=False,
    AssumeC=float(np.mean(C)),
    AssumeG=float(np.mean(g)),
    Verbose=False
)

print(f"  Stage 1 - Corrected ke: {ke_corrected:.3f} (error: {abs(ke_corrected - ke)/ke*100:.1f}%)")

# Stage 2: Fix ke and estimate others
(theta_twostage_corr,
 patient_params_corr,
 L0_twostage_corr,
 results_twostage_corr) = fcnEstimatePKPD_FixedKe_Optimized(
    L_obs, A_obs, patient_age, patient_sofa, t, parmsL, ke_corrected,
    MaxIterEM=20,
    Verbose=False
)

time_twostage_corr = time.perf_counter() - tic
theta_twostage_corr_full = np.concatenate([theta_twostage_corr[:6], np.array([ke_corrected])])
errors_twostage_corr = np.abs(theta_twostage_corr_full - true_params) / np.abs(true_params) * 100.0
mape_twostage_corr = float(np.mean(errors_twostage_corr))
corr_twostage_corr, rmse_twostage_corr, mae_twostage_corr = computeL0Metrics(L0_twostage_corr, L0_true)

print(f"  Overall MAPE: {mape_twostage_corr:.1f}%, L0 corr: {corr_twostage_corr:.3f}")
print(f"  Time: {time_twostage_corr:.1f} seconds\n")

# ------------------------------------------------------------
# 6. L PREDICTION ANALYSIS (using best method)
# ------------------------------------------------------------
print("[6/6] Analyzing L prediction accuracy...")
tic = time.perf_counter()

# Use corrected two-stage results (best method)
C_est = np.asarray(patient_params_corr['C_indiv']).reshape(-1)
g_est = np.asarray(patient_params_corr['g_indiv']).reshape(-1)
ke_est = float(ke_corrected)
L0_est = np.asarray(L0_twostage_corr)

# Predict L using TRUE L0 and ESTIMATED parameters
L_pred_trueL0 = np.zeros((N, T_len), dtype=float)
for i in range(N):
    X = np.zeros(T_len, dtype=float)
    for j in range(1, T_len):
        X[j] = ke_est * X[j - 1] + A_obs[i, j]
    for j in range(T_len):
        if X[j] > 0:
            sX = 1.0 - 1.0 / (((C_est[i] / X[j]) ** g_est[i]) + 1.0)
        else:
            sX = 1.0
        L_pred_trueL0[i, j] = L0_true[i, j] * sX

# Predict L using ESTIMATED L0 and ESTIMATED parameters
L_pred_estL0 = np.zeros((N, T_len), dtype=float)
for i in range(N):
    X = np.zeros(T_len, dtype=float)
    for j in range(1, T_len):
        X[j] = ke_est * X[j - 1] + A_obs[i, j]
    for j in range(T_len):
        if X[j] > 0:
            sX = 1.0 - 1.0 / (((C_est[i] / X[j]) ** g_est[i]) + 1.0)
        else:
            sX = 1.0
        L_pred_estL0[i, j] = L0_est[i, j] * sX

# Calculate prediction metrics
mape_L_trueL0 = np.zeros((N, 1), dtype=float)
mape_L_estL0 = np.zeros((N, 1), dtype=float)
corr_L_trueL0 = np.zeros((N, 1), dtype=float)
corr_L_estL0 = np.zeros((N, 1), dtype=float)

for i in range(N):
    valid = np.isfinite(L_obs[i, :]) & (L_obs[i, :] > 0)
    if np.sum(valid) > 10:
        mape_L_trueL0[i, 0] = np.mean(np.abs(L_pred_trueL0[i, valid] - L_obs[i, valid]) / L_obs[i, valid]) * 100.0
        mape_L_estL0[i, 0] = np.mean(np.abs(L_pred_estL0[i, valid] - L_obs[i, valid]) / L_obs[i, valid]) * 100.0
        # correlations
        x1, y1 = L_pred_trueL0[i, valid], L_obs[i, valid]
        x2, y2 = L_pred_estL0[i, valid], L_obs[i, valid]
        if np.std(x1) > 0 and np.std(y1) > 0:
            corr_L_trueL0[i, 0] = np.corrcoef(x1, y1)[0, 1]
        if np.std(x2) > 0 and np.std(y2) > 0:
            corr_L_estL0[i, 0] = np.corrcoef(x2, y2)[0, 1]

time_prediction = time.perf_counter() - tic
# match MATLAB's mean excluding outliers >100%
mask_true = mape_L_trueL0[:, 0] < 100
mask_est = mape_L_estL0[:, 0] < 100
print(f"  L prediction with true L0: MAPE = {np.mean(mape_L_trueL0[mask_true, 0]):.1f}%")
print(f"  L prediction with est L0:  MAPE = {np.mean(mape_L_estL0[mask_est, 0]):.1f}%")
print(f"  Time: {time_prediction:.1f} seconds\n")

# ------------------------------------------------------------
# COMPILE ALL RESULTS
# ------------------------------------------------------------
print("Compiling and saving results...")

results = {}

# Data info
results["N"] = N
results["T_len"] = T_len
results["dt"] = dt
results["t"] = t

# True values
results["true_params"] = true_params
results["param_names"] = np.array(param_names, dtype=object)
results["L0_true"] = L0_true
results["L_obs"] = L_obs
results["A_obs"] = A_obs
results["patient_age"] = patient_age
results["patient_sofa"] = patient_sofa
results["C_true"] = C_true
results["g_true"] = g_true
results["ke_true"] = ke

# Method 1: Joint
results["joint"] = {
    "theta": theta_joint,
    "patient_params": patient_params_joint,
    "L0_est": L0_joint,
    "errors": errors_joint,
    "mape": mape_joint,
    "L0_corr": corr_joint,
    "L0_rmse": rmse_joint,
    "time": time_joint,
}

# Method 2: Oracle
results["oracle"] = {
    "theta": theta_oracle_full,
    "patient_params": patient_params_oracle,
    "L0_est": L0_oracle,
    "errors": errors_oracle,
    "mape": mape_oracle,
    "L0_corr": corr_oracle,
    "L0_rmse": rmse_oracle,
    "time": time_oracle,
}

# Method 3: Two-stage raw
results["twostage_raw"] = {
    "theta": theta_twostage_raw_full,
    "patient_params": patient_params_raw,
    "L0_est": L0_twostage_raw,
    "errors": errors_twostage_raw,
    "mape": mape_twostage_raw,
    "L0_corr": corr_twostage_raw,
    "L0_rmse": rmse_twostage_raw,
    "ke_raw": ke_raw,
    "time": time_twostage_raw,
}

# Method 4: Two-stage corrected
results["twostage_corr"] = {
    "theta": theta_twostage_corr_full,
    "patient_params": patient_params_corr,
    "L0_est": L0_twostage_corr,
    "errors": errors_twostage_corr,
    "mape": mape_twostage_corr,
    "L0_corr": corr_twostage_corr,
    "L0_rmse": rmse_twostage_corr,
    "ke_corrected": ke_corrected,
    "time": time_twostage_corr,
}

# Add R2 fields if they exist on the returned results
if isinstance(results_twostage_corr, dict):
    if "R2_C" in results_twostage_corr:
        results["twostage_corr"]["R2_C"] = results_twostage_corr["R2_C"]
    if "R2_g" in results_twostage_corr:
        results["twostage_corr"]["R2_g"] = results_twostage_corr["R2_g"]
else:
    # If it's an object with attributes
    if hasattr(results_twostage_corr, "R2_C"):
        results["twostage_corr"]["R2_C"] = results_twostage_corr.R2_C
    if hasattr(results_twostage_corr, "R2_g"):
        results["twostage_corr"]["R2_g"] = results_twostage_corr.R2_g

# L prediction results
results["L_prediction"] = {
    "L_pred_trueL0": L_pred_trueL0,
    "L_pred_estL0": L_pred_estL0,
    "mape_L_trueL0": mape_L_trueL0,
    "mape_L_estL0": mape_L_estL0,
    "corr_L_trueL0": corr_L_trueL0,
    "corr_L_estL0": corr_L_estL0,
}

# Save results (MATLAB -v7.3 uses HDF5; scipy.savemat writes v5 .mat.
# If you need v7.3 specifically, use the 'hdf5storage' package.)
savemat("PKPD_estimation_results.mat", {"results": results}, do_compression=True)

# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
print("\n==========================================================")
print("SUMMARY OF ALL METHODS")
print("==========================================================\n")

print(f"{'Method':<25s} {'MAPE(%)':>10s} {'ke Err(%)':>10s} {'L0 Corr':>10s} {'Time(s)':>10s}")
print(f"{'-'*25:<25s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")

methods_summary = [
    ("Joint (Unfixed)", mape_joint, errors_joint[6], corr_joint, time_joint),
    ("Oracle (True ke)", mape_oracle, 0.0, corr_oracle, time_oracle),
    ("Two-Stage (Raw)", mape_twostage_raw, errors_twostage_raw[6], corr_twostage_raw, time_twostage_raw),
    ("Two-Stage (Corrected)", mape_twostage_corr, errors_twostage_corr[6], corr_twostage_corr, time_twostage_corr),
]

for name, mape_val, ke_err, corr_val, tim in methods_summary:
    print(f"{name:<25s} {mape_val:10.1f} {ke_err:10.1f} {corr_val:10.3f} {tim:10.1f}")

print("\nL PREDICTION ACCURACY (Two-Stage Corrected):")
print(f"  Using true L0:      {np.mean(mape_L_trueL0[mask_true, 0]):.1f}% MAPE")
print(f"  Using estimated L0: {np.mean(mape_L_estL0[mask_est, 0]):.1f}% MAPE")

print("\n==========================================================")
print("RESULTS SAVED TO: PKPD_estimation_results.mat")
print("Run a1_EvaluatePKPD_estimates.m for plots and detailed analysis")
print("==========================================================")
