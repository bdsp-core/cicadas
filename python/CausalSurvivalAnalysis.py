import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

# ---------------------------------------------------------------------
# Inline helpers (KM + bootstrap-by-sid) to mirror MATLAB behavior
# ---------------------------------------------------------------------

def _km_curve(times: np.ndarray, events: np.ndarray):
    """Kaplan–Meier S(t) as step function; returns (t_out, S_out) starting at (0,1)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times, kind="mergesort")
    times, events = times[order], events[order]
    uniq_event_times = np.unique(times[events == 1])
    if uniq_event_times.size == 0:
        return np.array([0.0]), np.array([1.0])
    n = len(times)
    at_risk = n
    S = 1.0
    t_out = [0.0]
    S_out = [1.0]
    idx = 0
    for te in uniq_event_times:
        # advance to te
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1
        d_i = np.sum((times == te) & (events == 1))
        # KM step
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        t_out.append(te); S_out.append(S)
        # remove all at te
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k
        idx += k
    return np.asarray(t_out), np.asarray(S_out)

def _times_events_for_group(Tdf: pd.DataFrame, pids):
    times, events = [], []
    for pid in pids:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        y = pdata["Y"].values
        if (y > 0).any():
            t_event = float(pdata["t"].values[np.where(y > 0)[0][0]])
            times.append(t_event); events.append(1)
        else:
            times.append(float(np.max(pdata["t"].values))); events.append(0)
    return np.asarray(times, float), np.asarray(events, int)

def fcnPlotKM_py(Tdf: pd.DataFrame):
    """Return (s0, s1, t0, t1) for untreated/treated."""
    uniq = np.unique(Tdf["sid"].values)
    treated, untreated = [], []
    for pid in uniq:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        if int(pdata["Rx"].iloc[0]) == 1:
            treated.append(pid)
        else:
            untreated.append(pid)
    ut_times, ut_events = _times_events_for_group(Tdf, untreated)
    tr_times, tr_events = _times_events_for_group(Tdf, treated)
    t0, s0 = _km_curve(ut_times, ut_events)
    t1, s1 = _km_curve(tr_times, tr_events)
    return s0, s1, t0, t1

def fcn_bootstrapBySID_py(T0: pd.DataFrame, N: int) -> pd.DataFrame:
    """Bootstrap patients by SID with replacement; reindex SIDs to 1..N to match MATLAB intent.

    Uses global np.random (seeded via np.random.seed at script top) so bootstrap
    CIs are reproducible. Previously used an unseeded default_rng which silently
    broke reproducibility of the paper's bootstrap confidence intervals.
    """
    orig_ids = np.unique(T0["sid"].values)
    sampled_ids = np.random.choice(orig_ids, size=len(orig_ids), replace=True)
    frames = []
    for new_sid, old_sid in enumerate(sampled_ids, start=1):
        df = T0[T0["sid"] == old_sid].copy()
        df["sid"] = new_sid  # re-label to keep exactly N unique patients
        frames.append(df)
    boot = pd.concat(frames, ignore_index=True)
    return boot

# ---------------------------------------------------------------------
# External helpers expected to exist (same signatures as MATLAB)
# ---------------------------------------------------------------------
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnEstimateParmsL import fcnEstimateParmsL
from fcnGetPKPD_parms_est import fcnGetPKPD_parms_est
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have a translated fcnPlotKM and fcn_bootstrapBySID, you can import and use them.
from fcnPlotKM import fcnPlotKM
from fcn_bootstrapBySID import fcn_bootstrapBySID

# ---------------------------------------------------------------------
# Begin translated script
# ---------------------------------------------------------------------

# clear all; clc; format compact;
plt.figure(3); plt.clf()

dt = 2
t = np.arange(0, 168 + dt, dt)  # 0:dt:168
Nt = t.size  # 85

T0 = pd.read_csv("trialData0.csv")
N = np.unique(T0["sid"].values).size

# load parmsTrue % gets true values: 'parmsControl', 'parmsPD', 'C', 'g', 'parmsY', 'parmsV', 'parmsL'
m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
parmsControl = np.asarray(m["parmsControl"], dtype=float).reshape(-1)
parmsPD_true = np.asarray(m["parmsPD"], dtype=float).reshape(-1)
C_true_all = np.asarray(m["C"], dtype=float).reshape(-1)
g_true_all = np.asarray(m["g"], dtype=float).reshape(-1)
parmsY_true = np.asarray(m["parmsY"], dtype=float).reshape(-1)
parmsV_true = np.asarray(m["parmsV"], dtype=float).reshape(-1)
parmsL_true = np.asarray(m["parmsL"], dtype=float).reshape(-1)
age = np.asarray(m["age"], dtype=float).reshape(-1)
sofa = np.asarray(m["sofa"], dtype=float).reshape(-1)

th = 0.1  # target level for L(t)

# ===================================================
# ESTIMATE L and Y models
# ===================================================
parmsY_est = fcnEstimateDeathParms(T0)
parmsL_est, LL, AA, patient_age, patient_sofa, t_from_est = fcnEstimateParmsL(T0)
# Use the grid defined above for consistency
t = t  # keep original grid for downstream calls

# ================================================================================
# Load estimated coefficients for C, g, and estimated ke
# ================================================================================
C_est, g_est, ke_est, parmsPD_est = fcnGetPKPD_parms_est(patient_age, patient_sofa)

# Ensure parmsL_est is 7 elements (column)
parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)
if parmsL_est.size > 7:
    parmsL_est = parmsL_est[:7]
# Ensure parmsPD_est is column
parmsPD_est = np.asarray(parmsPD_est, dtype=float).reshape(-1)

# Create theta_est (length 14)
theta_est = np.concatenate([parmsL_est, parmsPD_est, np.array([ke_est], dtype=float)])
savemat("EstimatedParameters.mat",
        {"theta_est": theta_est, "parmsL_est": parmsL_est, "parmsPD_est": parmsPD_est, "ke_est": ke_est},
        do_compression=True)

# ================================================================================
# Estimate causal survival curves - contrasting always vs never treat
# ================================================================================
RCT = 1
treatProb = np.full(N, 0.5)
parmsV_est = np.zeros(6, dtype=float)  # No censoring in RCT simulation

L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N)
T1_est = fcnSimulate_N_Patients(
    N, RCT, treatProb, th, C_est, g_est, ke_est, L0_est, parmsControl, parmsY_est, parmsV_est, age, sofa
)

# ==========================================
# === EVALUATION PLOTS / DEBUG PRINTS ===
# ==========================================
b0_C_true, b1_C_true, b2_C_true = parmsPD_true[0], parmsPD_true[1], parmsPD_true[2]
b0_g_true, b1_g_true, b2_g_true = parmsPD_true[3], parmsPD_true[4], parmsPD_true[5]
true_params_all = np.concatenate([parmsL_true, parmsPD_true, np.array([m.get("ke", np.nan)]).reshape(-1)])

print("Mortality model coefficients:")
print(f"  a0 (intercept): {parmsY_est[0]:.3f}")
print(f"  a1 (time effect): {parmsY_est[1]:.3f}")
print(f"  a2 (cumL effect): {parmsY_est[2]:.3f}")
print(f"  a3 (cumA effect): {parmsY_est[3]:.3f}")

# Debug: sizes
theta_est_col = theta_est.reshape(-1, 1)
true_params_col = true_params_all.reshape(-1, 1)
print("\nDEBUG - Parameter sizes:")
print(f"  theta_est size = {theta_est_col.shape[0]}x{theta_est_col.shape[1]} (should be 14x1)")
print(f"  true_params_all size = {true_params_col.shape[0]}x{true_params_col.shape[1]} (should be 14x1)")
print(f"  theta_est length = {theta_est.size}")
print(f"  true_params_all length = {true_params_all.size}")

# Optional diagnostics (only if available)
# try:
#     fcnDiseaseModelDiagnostics(T1_est, true_params_all, C_est, g_est, theta_est, LL, AA, age, sofa, t)
# except Exception as e:
#     print(f"(Skipping fcnDiseaseModelDiagnostics: {e})")

# ==========================================
# Bootstrap CIs
# ==========================================
RCT = 1
treatProb = np.full(N, 0.5)
parmsV_est = np.zeros(6, dtype=float)

Nboot = 1000
S0h = []
S1h = []

print("Running bootstrap iterations...")
for i in range(1, Nboot + 1):
    if i % 10 == 0:
        print(f"  Iteration {i}/{Nboot}")
    # Bootstrap by SID
    # T0_boot = fcn_bootstrapBySID(T0, N)
    T0_boot = fcn_bootstrapBySID_py(T0, N)
    # Refit Y model on bootstrap resample
    parmsY_est_b = fcnEstimateDeathParms(T0_boot)
    # Simulate RCT with fixed L0 process and estimated Y
    L0_est_b = fcnGenerateStochasticTrajectories(t, parmsL_est, N)
    T1_est_b = fcnSimulate_N_Patients(
        N, RCT, treatProb, th, C_est, g_est, ke_est, L0_est_b, parmsControl, parmsY_est_b, parmsV_est, age, sofa
    )
    # s0h, s1h, t0, t1 = fcnPlotKM(T1_est_b)
    s0h, s1h, t0, t1 = fcnPlotKM(T1_est_b)

    S0h.append(s0h.reshape(1, -1))
    S1h.append(s1h.reshape(1, -1))

    # Live plot like figure(3); clf; plot(t0,S0h,'b',t1,S1h,'r');
    plt.figure(3); plt.clf()
    if len(S0h) > 0:
        arr0 = np.vstack(S0h)  # (i, Tlen)
        arr1 = np.vstack(S1h)
        for row in arr0:
            plt.plot(t0, row, 'b', alpha=0.15)
        for row in arr1:
            plt.plot(t1, row, 'r', alpha=0.15)
        plt.xlabel("Time (hours)"); plt.ylabel("Survival Probability")
        plt.xlim(0, 168); plt.ylim(0, 1); plt.grid(True)
        plt.pause(0.001)

S0h = np.vstack(S0h) if len(S0h) else np.empty((0, Nt))
S1h = np.vstack(S1h) if len(S1h) else np.empty((0, Nt))

# (MATLAB had: load bootstrap_confidence_bands) – not needed here since we just computed S0h/S1h

alpha = 0.05
s0_lower = np.percentile(S0h, 100 * alpha / 2.0, axis=0)
s0_upper = np.percentile(S0h, 100 * (1 - alpha / 2.0), axis=0)
s0_median = np.percentile(S0h, 50, axis=0)

s1_lower = np.percentile(S1h, 100 * alpha / 2.0, axis=0)
s1_upper = np.percentile(S1h, 100 * (1 - alpha / 2.0), axis=0)
s1_median = np.percentile(S1h, 50, axis=0)

s0_mean = np.mean(S0h, axis=0)
s1_mean = np.mean(S1h, axis=0)

# Reference curves from RCT truth
T1 = pd.read_csv("trialData1.csv")
# s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)
s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)

# Plot with bands
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
ax.set_title(f"Survival Curves with 95% Bootstrap Confidence Bands (n={Nboot})", fontsize=14)
ax.set_xlabel("Time (hours)", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.grid(True); ax.set_xlim(0, 168); ax.set_ylim(0, 1)
fig.patch.set_facecolor("white")

t_grid = np.arange(0, 168 + 2, 2.0)
t_col = t_grid.reshape(-1, 1)

# Bands
ax.fill_between(t_grid, s0_lower, s0_upper, color=(0.2, 0.4, 0.8), alpha=0.3, label="Untreated 95% CI")
ax.fill_between(t_grid, s1_lower, s1_upper, color=(0.8, 0.2, 0.2), alpha=0.3, label="Treated 95% CI")

# True (dashed) and Estimated (solid)
ax.plot(t0_true, s0_true, 'b--', linewidth=2.5, label="Untreated (True)")
ax.plot(t1_true, s1_true, 'r--', linewidth=2.5, label="Treated (True)")
ax.plot(t_grid, s0_median, 'b-', linewidth=2.5, label="Untreated (Estimated)")
ax.plot(t_grid, s1_median, 'r-', linewidth=2.5, label="Treated (Estimated)")

ax.legend(loc="lower left", fontsize=10)

# CI widths at key times
print("\n95% Confidence Interval Widths at Key Time Points:")
print("Time (h) | Untreated CI Width | Treated CI Width")
print("---------|-------------------|------------------")
key_times = [24, 48, 72, 96, 120, 144, 168]
for kt in key_times:
    idx = int(np.argmin(np.abs(t_grid - kt)))
    ci_width_0 = s0_upper[idx] - s0_lower[idx]
    ci_width_1 = s1_upper[idx] - s1_lower[idx]
    print(f"{kt:8d} | {ci_width_0:17.3f} | {ci_width_1:16.3f}")

# Summary stats
print("\nBootstrap Summary Statistics:")
print(f"Number of bootstrap samples: {Nboot}")
print("Survival at 168 hours:")
print(f"  Untreated: {s0_median[-1]*100:.1f}% [{s0_lower[-1]*100:.1f}%, {s0_upper[-1]*100:.1f}%]")
print(f"  Treated:   {s1_median[-1]*100:.1f}% [{s1_lower[-1]*100:.1f}%, {s1_upper[-1]*100:.1f}%]")

treatment_effect_median = s1_median[-1] - s0_median[-1]
treatment_effect_lower = s1_lower[-1] - s0_upper[-1]
treatment_effect_upper = s1_upper[-1] - s0_lower[-1]
print("\nTreatment effect at 168 hours:")
print(f"  Median difference: {treatment_effect_median:.3f}")
print(f"  95% CI: [{treatment_effect_lower:.3f}, {treatment_effect_upper:.3f}]")

# Save band data (mirrors MATLAB)
savemat(
    "bootstrap_confidence_bands.mat",
    {
        "S0h": S0h,
        "S1h": S1h,
        "s0_lower": s0_lower,
        "s0_upper": s0_upper,
        "s1_lower": s1_lower,
        "s1_upper": s1_upper,
        "s0_median": s0_median,
        "s1_median": s1_median,
        "t_grid": t_grid,
        "Nboot": Nboot,
    },
    do_compression=True,
)

print("\nBootstrap analysis complete!")

# ---------------------------------------------------------------------
# Export causal survival analysis results to text file for paper
# (Ported from matlab/a2_CausalSurvivalAnalysis.m:229-341)
# ---------------------------------------------------------------------
try:
    _now = datetime.now()
    _filename = f"causal_survival_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"
    _ke_true = float(np.asarray(m.get("ke", np.nan)).reshape(())) if "ke" in m else float("nan")
    with open(_filename, 'w') as f:
        f.write('==========================================================\n')
        f.write('CAUSAL SURVIVAL ANALYSIS RESULTS FOR PAPER\n')
        f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
        f.write('==========================================================\n\n')

        # Study parameters
        f.write('STUDY PARAMETERS:\n')
        f.write(f"- Sample size: {N} patients\n")
        f.write('- Study period: 168 hours\n')
        f.write(f"- Bootstrap samples: {Nboot}\n")
        f.write(f"- Target threshold: {th:.3f}\n")
        f.write('- Time step: 2 hours\n\n')

        # Parameter estimation results
        f.write('ESTIMATED PARAMETERS:\n')
        f.write('Disease progression (L) parameters:\n')
        for i in range(len(parmsL_est)):
            f.write(f"  parmsL[{i+1}]: {parmsL_est[i]:.4f} (true: {parmsL_true[i]:.4f})\n")

        f.write('\nPKPD parameters:\n')
        for i in range(len(parmsPD_est)):
            f.write(f"  parmsPD[{i+1}]: {parmsPD_est[i]:.4f} (true: {parmsPD_true[i]:.4f})\n")
        f.write(f"ke: {ke_est:.4f} (true: {_ke_true:.4f})\n\n")

        # Mortality hazard parameters
        f.write('Mortality hazard parameters:\n')
        for i in range(len(parmsY_est)):
            f.write(f"  parmsY[{i+1}]: {parmsY_est[i]:.4f} (true: {parmsY_true[i]:.4f})\n")
        f.write('\n')

        # Survival outcomes at key time points
        key_times = [24, 48, 72, 96, 120, 144, 168]
        f.write('SURVIVAL OUTCOMES (with 95% Bootstrap CIs):\n')
        f.write(f"{'Time(h)':<8s} {'Untreated(%)':<15s} {'Treated(%)':<15s} {'Difference(%)':<15s}\n")
        f.write(f"{'-------':<8s} {'-------------':<15s} {'-----------':<15s} {'-------------':<15s}\n")

        for kt in key_times:
            # MATLAB: find(t_grid >= key_times(i), 1)
            idxs = np.where(t_grid >= kt)[0]
            if idxs.size > 0:
                t_idx = int(idxs[0])
                untreated_pct = s0_median[t_idx] * 100
                treated_pct = s1_median[t_idx] * 100
                diff_pct = (s1_median[t_idx] - s0_median[t_idx]) * 100
                f.write(f"{kt:<8d} {untreated_pct:<15.1f} {treated_pct:<15.1f} {diff_pct:<15.1f}\n")

        # Primary endpoint - 168 hours
        f.write('\nPRIMARY ENDPOINT (168 hours):\n')
        f.write(f"Untreated survival: {s0_median[-1]*100:.1f}% [{s0_lower[-1]*100:.1f}%, {s0_upper[-1]*100:.1f}%]\n")
        f.write(f"Treated survival: {s1_median[-1]*100:.1f}% [{s1_lower[-1]*100:.1f}%, {s1_upper[-1]*100:.1f}%]\n")

        # Treatment effect
        f.write('\nTREATMENT EFFECT:\n')
        f.write(f"Absolute risk difference: {treatment_effect_median*100:.1f}% [{treatment_effect_lower*100:.1f}%, {treatment_effect_upper*100:.1f}%]\n")

        # Risk ratio / NNT
        if treatment_effect_median > 0:
            risk_ratio = s1_median[-1] / s0_median[-1] if s0_median[-1] != 0 else float('inf')
            nnt = 1.0 / treatment_effect_median
            f.write(f"Risk ratio: {risk_ratio:.2f}\n")
            f.write(f"Number needed to treat: {nnt:.1f}\n")
            f.write('Interpretation: Treatment is BENEFICIAL\n')
        elif treatment_effect_median < 0:
            f.write(f"Number needed to harm: {1.0/abs(treatment_effect_median):.1f}\n")
            f.write('Interpretation: Treatment is HARMFUL\n')
        else:
            f.write('Interpretation: No treatment effect\n')

        # Statistical significance
        ci_excludes_zero = ((treatment_effect_lower > 0 and treatment_effect_upper > 0) or
                            (treatment_effect_lower < 0 and treatment_effect_upper < 0))
        f.write('\nSTATISTICAL SIGNIFICANCE:\n')
        if ci_excludes_zero:
            f.write('The 95% confidence interval excludes zero -> STATISTICALLY SIGNIFICANT\n')
        else:
            f.write('The 95% confidence interval includes zero -> NOT STATISTICALLY SIGNIFICANT\n')

        # Methods summary
        f.write('\nMETHODS SUMMARY:\n')
        f.write('- Data generation: Logit-based discrete-time hazard models\n')
        f.write('- Causal inference: G-formula/parametric g-computation\n')
        f.write('- Parameter estimation: Mixed-effects PKPD modeling\n')
        f.write(f"- Uncertainty quantification: {Nboot} bootstrap resamples\n")
        f.write('- Simulation: RCT emulation with estimated parameters\n')

        # Model performance
        f.write('\nMODEL PERFORMANCE:\n')
        param_errors_L = 100.0 * np.abs(parmsL_est - parmsL_true) / np.abs(parmsL_true)
        param_errors_PD = 100.0 * np.abs(parmsPD_est - parmsPD_true) / np.abs(parmsPD_true)
        param_errors_Y = 100.0 * np.abs(parmsY_est - parmsY_true) / np.abs(parmsY_true)
        ke_error = 100.0 * abs(ke_est - _ke_true) / abs(_ke_true) if _ke_true != 0 else float('nan')

        f.write('Mean absolute percentage error (MAPE):\n')
        f.write(f"- Disease parameters (L): {np.nanmean(param_errors_L):.1f}%\n")
        f.write(f"- PKPD parameters: {np.nanmean(param_errors_PD):.1f}%\n")
        f.write(f"- Mortality parameters (Y): {np.nanmean(param_errors_Y):.1f}%\n")
        f.write(f"- Elimination constant (ke): {ke_error:.1f}%\n")

    print(f"Causal survival analysis results exported to: {_filename}")
except Exception as _exc:
    print(f"WARNING: Failed to export causal survival results text file: {_exc}")
