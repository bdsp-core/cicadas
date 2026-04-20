# -*- coding: utf-8 -*-
# %% get 3 curves — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

# ---------------------------------------------------------------------
# KM helpers (to stand in for fcnPlotKM)
# ---------------------------------------------------------------------
def _km_curve(times: np.ndarray, events: np.ndarray):
    """Kaplan–Meier S(t) step function; returns (t_out, S_out) starting at (0,1)."""
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
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1
        d_i = np.sum((times == te) & (events == 1))
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        t_out.append(te); S_out.append(S)
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
    """Return (s0, s1, t0, t1) for untreated/treated groups via KM."""
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

# ---------------------------------------------------------------------
# External helpers (translated elsewhere)
# ---------------------------------------------------------------------
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnEstimateParmsL import fcnEstimateParmsL
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have a direct translation of fcnPlotKM, you can import and use it instead:
# from fcnPlotKM import fcnPlotKM  # returns (s0, s1, t0, t1)

# ---------------------------------------------------------------------
# Script body (mirrors MATLAB)
# ---------------------------------------------------------------------
# clear all; clc; format compact;
plt.figure(3); plt.clf()

dt = 2.0
t = np.arange(0.0, 168.0 + dt, dt)  # 0:dt:168
Nt = t.size

T0 = pd.read_csv("trialData0.csv")
N = np.unique(T0["sid"].values).size

# load parmsTrue
m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
parmsControl = np.asarray(m["parmsControl"], dtype=float).reshape(-1)
parmsPD = np.asarray(m["parmsPD"], dtype=float).reshape(-1)
C = np.asarray(m["C"], dtype=float).reshape(-1)
g = np.asarray(m["g"], dtype=float).reshape(-1)
parmsY = np.asarray(m["parmsY"], dtype=float).reshape(-1)
parmsV = np.asarray(m["parmsV"], dtype=float).reshape(-1)
parmsL = np.asarray(m["parmsL"], dtype=float).reshape(-1)
age = np.asarray(m["age"], dtype=float).reshape(-1)
sofa = np.asarray(m["sofa"], dtype=float).reshape(-1)
ke = float(np.asarray(m["ke"]).reshape(())) if "ke" in m else np.nan

th = np.array([0.0, 0.02, 0.1, 0.8], dtype=float)

# Estimate parameters
parmsY_est = fcnEstimateDeathParms(T0)
parmsL_est, LL, AA, patient_age, patient_sofa, t_local = fcnEstimateParmsL(T0)

# Setup for RCT simulation
RCT = 1
treatProb = np.full(N, 0.5)
parmsV_est = np.zeros(6, dtype=float)
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

# Local arrays
ATEest = np.zeros(len(th), dtype=float)
ATEref = np.zeros(len(th), dtype=float)
S1est = np.zeros((len(th), Nt), dtype=float)
S0est = np.zeros((len(th), Nt), dtype=float)
S1ref = np.zeros((len(th), Nt), dtype=float)
S0ref = np.zeros((len(th), Nt), dtype=float)

for i, thi in enumerate(th):
    T_ref = fcnSimulate_N_Patients(N, RCT, treatProb, thi, C, g, ke, L0, parmsControl, parmsY, parmsV_est, age, sofa)
    T_est = fcnSimulate_N_Patients(N, RCT, treatProb, thi, C, g, ke, L0, parmsControl, parmsY_est, parmsV_est, age, sofa)

    # s0,s1 from ref; s0_est,s1_est from est
    # If you have fcnPlotKM, call it; otherwise use fcnPlotKM_py:
    # s0, s1, _, _ = fcnPlotKM(T_ref); s0_est, s1_est, _, _ = fcnPlotKM(T_est)
    s0, s1, _, _ = fcnPlotKM_py(T_ref)
    s0_est, s1_est, _, _ = fcnPlotKM_py(T_est)

    # Align to common grid length Nt (defensive; KM returns step times)
    def _pad_to_nt(s):
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size >= Nt:
            return s[:Nt]
        return np.pad(s, (0, Nt - s.size), constant_values=np.nan)

    s0g = _pad_to_nt(s0); s1g = _pad_to_nt(s1)
    s0eg = _pad_to_nt(s0_est); s1eg = _pad_to_nt(s1_est)

    ATEest[i] = s1eg[~np.isnan(s1eg)][-1] - s0eg[~np.isnan(s0eg)][-1]
    ATEref[i] = s1g[~np.isnan(s1g)][-1] - s0g[~np.isnan(s0g)][-1]

    S1est[i, :] = s1eg
    S0est[i, :] = s0eg
    S1ref[i, :] = s1g
    S0ref[i, :] = s0g

# Save like MATLAB "save ThreeCurves"
savemat(
    "ThreeCurves.mat",
    {
        "th": th,
        "ATEest": ATEest,
        "ATEref": ATEref,
        "S1est": S1est,
        "S0est": S0est,
        "S1ref": S1ref,
        "S0ref": S0ref,
        "t": t,
        "Nt": Nt,
    },
    do_compression=True,
)

# %%
# Plot (fixed minor indexing typo from MATLAB: S0ref/S0est should use [i, :])
plt.figure(1); plt.clf()
for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    ax.plot(t, S1ref[i, :], 'r--', label='Treated (True)')
    ax.plot(t, S0ref[i, :], 'b--', label='Untreated (True)')
    ax.plot(t, S1est[i, :], 'r', label='Treated (Est)')
    ax.plot(t, S0est[i, :], 'b', label='Untreated (Est)')
    ax.set_xlim(0, 168); ax.set_ylim(0, 1)
    ax.grid(True)
    if i == 0:
        ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
# plt.show()  # optional

# ---------------------------------------------------------------------
# Export treatment targets comparison results to text file for paper
# (Ported from matlab/a3_ThreeTreatmentTargets.m:46-242)
# ---------------------------------------------------------------------
try:
    _now = datetime.now()
    _filename = f"treatment_targets_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(_filename, 'w') as f:
        f.write('==========================================================\n')
        f.write('TREATMENT TARGETS COMPARISON RESULTS FOR PAPER\n')
        f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
        f.write('==========================================================\n\n')

        # Study parameters
        f.write('STUDY PARAMETERS:\n')
        f.write(f"- Sample size: {N} patients\n")
        f.write('- Study period: 168 hours\n')
        f.write(f"- Time step: {int(dt)} hours\n")
        f.write(f"- Number of treatment targets tested: {len(th)}\n")
        _th_str = ' '.join([f"{x:.2f}" for x in th])
        f.write(f"- Target thresholds: [{_th_str} ]\n\n")

        # Treatment target descriptions
        f.write('TREATMENT TARGET DESCRIPTIONS:\n')
        target_descriptions = [
            ('No treatment (th = 0.00)', 'No active disease suppression'),
            ('Mild treatment (th = 0.02)', 'Minimal disease suppression'),
            ('Standard treatment (th = 0.10)', 'Moderate disease suppression'),
            ('Aggressive treatment (th = 0.80)', 'Intensive disease suppression'),
        ]
        for i in range(len(th)):
            if i < len(target_descriptions):
                d1, d2 = target_descriptions[i]
                f.write(f"  Target {i+1}: {d1} - {d2}\n")
            else:
                f.write(f"  Target {i+1}: th = {th[i]:.2f}\n")
        f.write('\n')

        # Parameter estimation accuracy
        param_errors_L = 100.0 * np.abs(parmsL_est - parmsL) / np.abs(parmsL)
        param_errors_Y = 100.0 * np.abs(parmsY_est - parmsY) / np.abs(parmsY)

        f.write('PARAMETER ESTIMATION ACCURACY:\n')
        f.write('Disease progression (L) parameters:\n')
        param_names_L = ['growth_rate', 'peak_height', 'alpha', 'decay_rate',
                         'sigma_early', 'sigma_late', 'sigma_transition']
        n_L = min(len(parmsL_est), len(param_names_L))
        for i in range(n_L):
            f.write(f"  {param_names_L[i]}: {parmsL_est[i]:.4f} (true: {parmsL[i]:.4f}, error: {param_errors_L[i]:.1f}%)\n")

        f.write('\nMortality hazard (Y) parameters:\n')
        for i in range(len(parmsY_est)):
            f.write(f"  a{i}: {parmsY_est[i]:.4f} (true: {parmsY[i]:.4f}, error: {param_errors_Y[i]:.1f}%)\n")
        f.write(f"Overall MAPE - L parameters: {np.nanmean(param_errors_L):.1f}%\n")
        f.write(f"Overall MAPE - Y parameters: {np.nanmean(param_errors_Y):.1f}%\n\n")

        # Treatment effects comparison
        f.write('TREATMENT EFFECTS COMPARISON (ATE at 168 hours):\n')
        f.write(f"{'Target':<25s} {'True ATE':<12s} {'Est ATE':<12s} {'Bias':<12s} {'Bias(%)':<12s}\n")
        f.write(f"{'-'*25:<25s} {'-'*12:<12s} {'-'*12:<12s} {'-'*12:<12s} {'-'*12:<12s}\n")

        for i in range(len(th)):
            bias_abs = ATEest[i] - ATEref[i]
            bias_pct = 100.0 * bias_abs / abs(ATEref[i]) if ATEref[i] != 0 else float('nan')
            f.write(f"{'th = ' + f'{th[i]:.2f}':<25s} {ATEref[i]:+11.3f} {ATEest[i]:+11.3f} {bias_abs:+11.3f} {bias_pct:+11.1f}\n")

        # Survival outcomes at 168 hours
        f.write('\nSURVIVAL OUTCOMES AT 168 HOURS:\n')
        f.write(f"{'Target':<15s} {'Untrt(True)':<15s} {'Treat(True)':<15s} {'Untrt(Est)':<15s} {'Treat(Est)':<15s}\n")
        f.write(f"{'-'*15:<15s} {'-'*15:<15s} {'-'*15:<15s} {'-'*15:<15s} {'-'*15:<15s}\n")

        for i in range(len(th)):
            # Use last valid value (MATLAB uses end index)
            def _last_val(arr):
                a = np.asarray(arr, dtype=float)
                mask = ~np.isnan(a)
                return a[mask][-1] if mask.any() else float('nan')
            s0r = _last_val(S0ref[i, :])
            s1r = _last_val(S1ref[i, :])
            s0e = _last_val(S0est[i, :])
            s1e = _last_val(S1est[i, :])
            f.write(f"{'th = ' + f'{th[i]:.2f}':<15s} {s0r*100:<14.1f}% {s1r*100:<14.1f}% {s0e*100:<14.1f}% {s1e*100:<14.1f}%\n")

        # Optimal treatment target
        optimal_idx_true = int(np.argmax(ATEref))
        optimal_idx_est = int(np.argmax(ATEest))
        max_ate_true = ATEref[optimal_idx_true]
        max_ate_est = ATEest[optimal_idx_est]

        f.write('\nOPTIMAL TREATMENT TARGET ANALYSIS:\n')
        f.write('Based on TRUE parameters:\n')
        f.write(f"  Optimal target: th = {th[optimal_idx_true]:.2f}\n")
        f.write(f"  Maximum ATE: {max_ate_true:.3f} ({max_ate_true*100:.1f}% survival benefit)\n")
        def _last_val_row(row):
            a = np.asarray(row, dtype=float)
            mask = ~np.isnan(a)
            return a[mask][-1] if mask.any() else float('nan')
        f.write(f"  Survival rates: Untreated {_last_val_row(S0ref[optimal_idx_true, :])*100:.1f}%, Treated {_last_val_row(S1ref[optimal_idx_true, :])*100:.1f}%\n")

        f.write('\nBased on ESTIMATED parameters:\n')
        f.write(f"  Optimal target: th = {th[optimal_idx_est]:.2f}\n")
        f.write(f"  Maximum ATE: {max_ate_est:.3f} ({max_ate_est*100:.1f}% survival benefit)\n")
        f.write(f"  Survival rates: Untreated {_last_val_row(S0est[optimal_idx_est, :])*100:.1f}%, Treated {_last_val_row(S1est[optimal_idx_est, :])*100:.1f}%\n")

        # Agreement analysis
        target_agreement = (optimal_idx_true == optimal_idx_est)
        f.write('\nTARGET SELECTION AGREEMENT:\n')
        if target_agreement:
            f.write('+ TRUE and ESTIMATED parameters identify the same optimal target\n')
            f.write('  -> Parameter estimation is sufficient for treatment optimization\n')
        else:
            f.write('x TRUE and ESTIMATED parameters identify different optimal targets\n')
            f.write('  -> Parameter estimation bias affects treatment optimization\n')
            f.write(f"  True optimal: th = {th[optimal_idx_true]:.2f}, Estimated optimal: th = {th[optimal_idx_est]:.2f}\n")

        # Bias analysis across targets
        f.write('\nBIAS ANALYSIS ACROSS TARGETS:\n')
        diffs = np.abs(ATEest - ATEref)
        mean_bias = float(np.mean(diffs))
        max_bias = float(np.max(diffs))
        max_bias_idx = int(np.argmax(diffs))

        f.write(f"Average absolute bias: {mean_bias:.3f} ({mean_bias*100:.1f}% survival difference)\n")
        f.write(f"Maximum absolute bias: {max_bias:.3f} at th = {th[max_bias_idx]:.2f}\n")

        # Trend
        if len(th) > 2:
            idx_vec = np.arange(1, len(th) + 1, dtype=float)
            # Pearson correlation
            if np.std(idx_vec) > 0 and np.std(diffs) > 0:
                bias_trend = float(np.corrcoef(idx_vec, diffs)[0, 1])
            else:
                bias_trend = 0.0
            f.write(f"Bias vs target aggressiveness correlation: {bias_trend:.3f}\n")
            if bias_trend > 0.5:
                f.write('  -> Bias INCREASES with more aggressive targets\n')
            elif bias_trend < -0.5:
                f.write('  -> Bias DECREASES with more aggressive targets\n')
            else:
                f.write('  -> Bias is relatively CONSTANT across targets\n')

        # Clinical implications
        f.write('\nCLINICAL IMPLICATIONS:\n')

        beneficial_targets_true = int(np.sum(ATEref > 0))
        beneficial_targets_est = int(np.sum(ATEest > 0))
        harmful_targets_true = int(np.sum(ATEref < 0))
        harmful_targets_est = int(np.sum(ATEest < 0))

        f.write('Treatment benefit assessment:\n')
        f.write(f"  TRUE parameters: {beneficial_targets_true} beneficial, {harmful_targets_true} harmful targets\n")
        f.write(f"  ESTIMATED parameters: {beneficial_targets_est} beneficial, {harmful_targets_est} harmful targets\n")
        if beneficial_targets_true == beneficial_targets_est:
            f.write('  + Parameter estimation correctly identifies treatment benefit\n')
        else:
            f.write('  x Parameter estimation misclassifies treatment benefit\n')

        # Safety considerations
        if np.any(ATEref < 0):
            harmful_idx = np.where(ATEref < 0)[0]
            f.write('\nSAFETY CONSIDERATIONS:\n')
            for i in harmful_idx:
                f.write(f"  !!! th = {th[i]:.2f} appears HARMFUL (ATE = {ATEref[i]:.3f})\n")
                if ATEest[i] > 0:
                    f.write(f"     But estimated as beneficial (ATE = {ATEest[i]:.3f}) - DANGEROUS!\n")

        # Recommendations
        f.write('\nRECOMMENDATIONS:\n')
        if target_agreement and mean_bias < 0.05:
            f.write('+ PROCEED with treatment optimization using estimated parameters\n')
            f.write('  - Parameter estimation is sufficiently accurate\n')
            f.write('  - Optimal target correctly identified\n')
            f.write(f"  - Recommended target: th = {th[optimal_idx_est]:.2f}\n")
        else:
            f.write('!!! CAUTION required for treatment optimization\n')
            if not target_agreement:
                f.write('  - Parameter estimation affects optimal target selection\n')
                f.write(f"  - Consider sensitivity analysis around th = {th[optimal_idx_true]:.2f}\n")
            if mean_bias >= 0.05:
                f.write('  - High bias in treatment effect estimation\n')
                f.write('  - Improve parameter estimation before optimization\n')

        f.write('\nGeneral recommendations:\n')
        f.write('- Validate parameter estimates with independent data\n')
        f.write('- Consider dose-response relationships\n')
        f.write('- Monitor for unintended consequences\n')
        f.write('- Use adaptive treatment strategies\n')

        # Technical details
        f.write('\nTECHNICAL DETAILS:\n')
        f.write('- Simulation: RCT emulation with 50% treatment probability\n')
        f.write('- Endpoint: Survival probability at 168 hours\n')
        f.write('- Target mechanism: PI controller with variable threshold\n')
        f.write('- Parameter source: Estimated from observational data\n')
        f.write('- Comparison: True vs estimated parameter performance\n')

    print(f"Treatment targets comparison results exported to: {_filename}")
except Exception as _exc:
    print(f"WARNING: Failed to export treatment targets results text file: {_exc}")
